#!/usr/bin/env python

"""
The module using a base path as guider searches all possible paths around the vehicle toward to base path and finds the
best path for the car to follow
Module functions:
1) List of possible paths : A list of candidate paths for the vehicle to follow
2) Objects around the car: The position and orientation ot objects like other cars and pedestrian relative to our car


"""

# ==============================================================================
# --General imports ------------------------------------------------------------
# ==============================================================================
import sys
sys.path.append('./DynamicPolynomialPlanner')
from DynamicPolynomialPlanner import frenet_paths
from DynamicPolynomialPlanner import cubic_spline_planner
from evaluation import *
from time import sleep
import numpy as np
import math
import copy

# ==============================================================================
# -- ROS imports ---------------------------------------------------------------
# ==============================================================================
import rospy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from perception.msg import Object
from route_planner.srv import RoutePath, RoutePathRequest, RoutePathResponse
from perception.msg import ObjectsList
from perception.msg import Waypoint
from perception.msg import WaypointsList
from perception.msg import TrajectoriesList
from perception.msg import Lane
from maneuver_generator.msg import ManeuverDataROS
from local_path_planner.msg import LocalOptimalPath


# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================

# Global variables
MANEUVERS_TO_EXCLUDE_VEHICLES = ["VehicleFollow"]
END_PATH_OFFSET = 50  # Number of waypoints from the end of the path to consider that vehicle reaches the end
GLOBAL_DRAW = True


def generate_target_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp, s


def calc_min_lateral_dist(x_base, y_base, x_v, y_v, x0=0, offset=1000000, step=1):
    if len(x_base) != len(y_base):
        return None
    t1 = x0 - offset if (x0 - offset) > 0 else 0
    t2 = x0 + offset if (x0 + offset) < (len(x_base) - 1) else len(x_base)

    min_dist = float("inf")
    index = x0
    for i in range(0, t2 - t1, step):
        dist = math.sqrt((x_base[t1 + i] - x_v) ** 2 + (y_base[t1 + i] - y_v) ** 2)
        if min_dist > dist:
            min_dist = dist
            index = t1 + i

    return min_dist, index


class ManeuverDataLocPlan:

    def __init__(self, left_road_width=-5, right_road_width=5, lateral_offset=None, speed=8.34, path_number=11, from_time=2.0,
                 to_time=3.0, time_step=1.0, dt=0.3, direct_control=False, maneuver_type="Other"):
        self.maneuver_type = maneuver_type
        self.dt = dt
        # Time for traveling distance
        self.from_time = from_time
        self.to_time = to_time
        self.time_sample_step = time_step
        self.direct_control = direct_control
        # Position
        self.target_lateral_offset = lateral_offset
        self.left_road_width = left_road_width  # maximum road width to the left [m] <-- Maneuver
        self.right_road_width = right_road_width  # maximum road width to the right [m] <-- Maneuver
        self.num_of_paths_samples = path_number  # number of alternative paths <-- Compromise
        # Speed
        self.target_speed = speed  # target speed [m/s] <-- Maneuver
        self.sampling_length = 0.15*self.target_speed  # target speed sampling length [m/s] <-- Compromise
        self.num_of_speed_samples = 1  # sampling number of target speed <-- Compromise

    def get_lateral_point_list(self):
        return [self.left_road_width + i * (self.right_road_width - self.left_road_width) /
                (self.num_of_paths_samples - 1) for i in range(self.num_of_paths_samples)]

    def get_speed_points_list(self):
        return [self.target_speed - i * self.sampling_length for i in range(self.num_of_speed_samples + 1)] + \
                       [self.target_speed + i * self.sampling_length for i in range(1, self.num_of_speed_samples + 1)]

    def get_time_points_list(self):
        return [self.from_time + i * self.time_sample_step for i in range(int((self.to_time - self.from_time)
                                                                              / self.time_sample_step) + 1)]


class LocalPathPlanner:

    def __init__(self):
        self.vehicles = []
        self.pedestrians = []
        self.route_path = []
        self.candidate_paths = []
        self.client_get_route_path()
        self.maneuver_data = None
        self.stop_vehicle = False
        self.emergency_stop = False
        # Base path spline values initialization
        self.tx, self.ty, self.t_yaw, self.tc, self.t_csp, self.t_s = generate_target_course(
            [w.x for w in self.route_path], [w.y for w in self.route_path])
        self.loc_index = 0  # Vehicle location relative to base path
        self.KD = 1.0  # weight for distance of the candidate path from base path
        # --- ROS ---
        rospy.init_node('Local_path_planner_node', anonymous=True)
        self.ego_vehicle = Object()
        self.current_lane = Lane()
        self.subscriber_ego_vehicle = rospy.Subscriber("ego_vehicle_msg", Object, self.callback_ego_vehicle, queue_size=1)
        self.subscriber_vehicles_list = rospy.Subscriber('vehicles_list_msg', ObjectsList, self.callback_vehicles_list, queue_size=1)
        self.subscriber_pedestrians_list = rospy.Subscriber('pedestrians_list_msg', ObjectsList, self.callback_pedestrians_list, queue_size=1)
        self.subscriber_maneuver_data = rospy.Subscriber('maneuver_data_msg', ManeuverDataROS, self.callback_maneuver_data, queue_size=1)
        self.subscriber_current_lane = rospy.Subscriber('current_lane_information_msg', Lane, self.callback_current_lane, queue_size=1)
        self.pub_optimal_local_path = rospy.Publisher('optimal_local_path_msg', LocalOptimalPath, queue_size=1)
        self.optimal_path = self.init_optimal_path()
        # Draw trajectories
        if GLOBAL_DRAW:
            self.pub_candidate_paths_draw = rospy.Publisher('candidate_paths_draw_msg', TrajectoriesList, queue_size=1)

    def init_optimal_path(self):
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        # Optimal path initialization
        # Vehicle position relative to base path curve(to the right or to the left)
        Ax, Ay = self.tx[0], self.ty[0]
        Bx, By = self.tx[3], self.ty[3]
        curve_side = np.sign((Bx - Ax) * (ego_vehicle.y - Ay) - (By - Ay) * (ego_vehicle.x - Ax))
        curve_side = 1.0 if curve_side == 0.0 else curve_side
        # initial state
        c_speed = ego_vehicle.speed / 3.6  # current speed [m/s]
        dist_from_route = math.hypot(self.route_path[0].y - ego_vehicle.y,
                                     self.route_path[0].x - ego_vehicle.x)
        c_d = curve_side * dist_from_route  # current lateral position [m]
        c_d = 0.1 if c_d < 0.1 else c_d
        c_d_d = 0.0  # current lateral speed [m/s]
        c_d_dd = 0.0  # current lateral acceleration [m/s]
        s0 = 0.0  # current course position
        fn_paths = frenet_paths.frenet_planning(self.t_csp, s0, c_speed, c_d, c_d_d, c_d_dd)
        d = close_to_base_path(fn_paths)
        return fn_paths[d.index(max(d))]

    def find_optimal_local_path(self):
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        maneuver_data = copy.deepcopy(self.maneuver_data)
        # Localization
        dist, self.loc_index = calc_min_lateral_dist(x_base=self.tx, y_base=self.ty, x_v=ego_vehicle.x,
                                                     y_v=ego_vehicle.y, x0=self.loc_index, offset=400)
        # Condition if vehicle reaches the end of the path
        if (len(self.tx) - self.loc_index) < END_PATH_OFFSET:
            self.publish_optimal_local_path()
            return None
        # Vehicle position relative to curve
        Ax, Ay = self.tx[self.loc_index], self.ty[self.loc_index]
        Bx, By = self.tx[self.loc_index + 3], self.ty[self.loc_index + 3]
        curve_side = np.sign((Bx - Ax) * (ego_vehicle.y - Ay) - (By - Ay) * (ego_vehicle.x - Ax))
        curve_side = 1 if curve_side == 0 else curve_side
        # Position relative to previous path
        id_next = 1
        for j in range(len(self.optimal_path.s)):
            if self.optimal_path.s[j] > self.t_s[self.loc_index]:
                id_next = j
                break
        s0 = self.t_s[self.loc_index]
        c_d = curve_side * dist
        c_d_d = 0.0  # self.optimal_path.d_d[id_next]
        c_d_dd = 0.0  # self.optimal_path.d_dd[id_next]
        c_speed = self.optimal_path.s_d[id_next]
        stop_flag = False
        if maneuver_data is not None:
            if maneuver_data.target_speed < 1:
                maneuver_data.target_speed = 3
                stop_flag = True
        self.candidate_paths = frenet_paths.frenet_planning(self.t_csp, s0, c_speed, c_d, c_d_d, c_d_dd, maneuver_data)
        if stop_flag:
            for i in range(len(self.candidate_paths)):
                self.candidate_paths[i].s_d = [0]*len(self.candidate_paths[i].s_d)
        self.evaluate_paths()
        self.publish_optimal_local_path()
        if GLOBAL_DRAW:
            self.publish_candidate_paths_draw()
        '''
        # Draw paths
        if GLOBAL_DRAW:
            all_paths = []
            for j in range(len(candidate_paths)):
                all_paths.append([])
                for ii in range(len(candidate_paths[j].x)):
                    all_paths[j].append(carla.Location(x=candidate_paths[j].x[ii], y=candidate_paths[j].y[ii]))
            self.local_map.draw_paths(paths=all_paths, life_time=1.2, color=[250, 0, 0], same_color=True)
            # self.local_map.draw_paths(paths=[self.base_path], life_time=1.2, color=[0, 0, 250], same_color=True)'''

    def evaluate_paths(self, lane_borders=None):
        self.emergency_stop = False
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        pedestrians = copy.deepcopy(self.pedestrians)
        vehicles = copy.deepcopy(self.vehicles)
        maneuver_data = copy.deepcopy(self.maneuver_data)
        lateral_offset_goal = 0.0 if maneuver_data is None else maneuver_data.target_lateral_offset
        front_vehicle_id = self.current_lane.front_vehicle.object_id
        front_front_vehicle_id = self.current_lane.front_front_vehicle.object_id
        if maneuver_data is not None:
            if maneuver_data.maneuver_type in MANEUVERS_TO_EXCLUDE_VEHICLES:
                vehicles = [obj_i for obj_i in vehicles if obj_i.object_id != front_vehicle_id and obj_i.object_id != front_front_vehicle_id]

        # Vehicles collision
        W = []
        if len(vehicles) != 0:
            d1, vehicle_collision = far_away_from_objects(self.candidate_paths, vehicles, ego_vehicle)
        else:
            d1 = [1.0]*len(self.candidate_paths)
            vehicle_collision = False
        d2 = close_to_lateral_offset_target(self.candidate_paths, lateral_offset_goal)
        d3 = path_close_to_previous_one(self.candidate_paths, self.optimal_path)

        # Pedestrians collision
        if len(pedestrians) != 0:
            _, stop_vehicle_pedestrian = far_away_from_objects(self.candidate_paths, pedestrians, ego_vehicle)
        else:
            stop_vehicle_pedestrian = False
        self.stop_vehicle = vehicle_collision or stop_vehicle_pedestrian

        if self.stop_vehicle and ego_vehicle.speed < 1:
            sleep(0.2)
            for i in range(len(self.candidate_paths)-1):
                W.append(d2[i])
        else:
            for i in range(len(self.candidate_paths) - 1):
                W.append(self.KD * d2[i] + 0.2 * self.KD * d3[i])

        # Return best path, check if collides with any object
        maxW = max(W)
        while not rospy.is_shutdown():
            maxW = max(W)
            best_index = W.index(maxW)
            if d1[best_index] != 0.0:
                break
            else:
                if len(W) > 1:
                    W.pop(best_index)
                else:
                    self.stop_vehicle = True
                    self.optimal_path.s_d = [0] * len(self.optimal_path.s_d)
                    return False
        self.optimal_path = self.candidate_paths[W.index(maxW)]
        _, pedestrian_collision = far_away_from_objects([self.optimal_path], pedestrians, ego_vehicle)
        if pedestrian_collision:
            print("stop pedestrian collision")
        if self.stop_vehicle or pedestrian_collision:
            print("----------->  Vehicle had to stop")
            self.emergency_stop = True
            self.optimal_path.s_d = [0]*len(self.optimal_path.s_d)

    # -------- ROS functions ---------
    def callback_ego_vehicle(self, ros_data):
        self.ego_vehicle = ros_data
        #rospy.loginfo(ros_data)

    def callback_vehicles_list(self, ros_data):
        self.vehicles = ros_data.objects_list
        #rospy.loginfo(ros_data)

    def callback_pedestrians_list(self, ros_data):
        self.pedestrians = ros_data.objects_list
        #rospy.loginfo(ros_data)

    def callback_current_lane(self, ros_data):
        self.current_lane = ros_data
        #rospy.loginfo(ros_data)

    def callback_maneuver_data(self, ros_data):
        self.maneuver_data = ManeuverDataLocPlan(left_road_width=ros_data.left_road_width, right_road_width=ros_data.right_road_width,
                                                 lateral_offset=ros_data.target_lateral_offset, speed=ros_data.target_speed,
                                                 path_number=ros_data.num_of_paths_samples, from_time=ros_data.from_time, to_time=ros_data.to_time,
                                                 time_step=ros_data.time_sample_step, dt=ros_data.dt, direct_control=ros_data.direct_control,
                                                 maneuver_type=ros_data.maneuver_type[0])
        #rospy.loginfo(ros_data)

    def publish_optimal_local_path(self):
        pub = self.pub_optimal_local_path
        optimal_path = LocalOptimalPath()
        optimal_path.x = self.optimal_path.x
        optimal_path.y = self.optimal_path.y
        optimal_path.yaw = self.optimal_path.yaw
        optimal_path.s_d = self.optimal_path.s_d
        optimal_path.emergency_stop = self.emergency_stop
        if self.maneuver_data is not None:
            optimal_path.direct_control = self.maneuver_data.direct_control
            optimal_path.direct_target_speed = self.maneuver_data.target_speed
        else:
            optimal_path.direct_control = False
            optimal_path.direct_target_speed = 0
        #rospy.loginfo(optimal_path)
        pub.publish(optimal_path)

    def publish_candidate_paths_draw(self):
        pub = self.pub_candidate_paths_draw
        candidate_paths = []
        for fn_path in self.candidate_paths:
            trajectory = []
            for i in range(len(fn_path.yaw)):
                ros_w = Waypoint()
                ros_w.x = fn_path.x[i]
                ros_w.y = fn_path.y[i]
                ros_w.yaw = fn_path.yaw[i]
                trajectory.append(ros_w)
            ros_trajectory = WaypointsList()
            ros_trajectory.waypoints_list = trajectory
            candidate_paths.append(ros_trajectory)
        ros_candidate_paths = TrajectoriesList()
        ros_candidate_paths.trajectories_list = candidate_paths
        #rospy.loginfo(ros_candidate_paths)
        pub.publish(ros_candidate_paths)

    def client_get_route_path(self):
        rospy.wait_for_service('route_path_srv')
        try:
            route_path_srv = rospy.ServiceProxy('route_path_srv', RoutePath)
            resp1 = route_path_srv(0)
            #rospy.loginfo(resp1)
            self.route_path = resp1.route_path
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


def main():
    #sleep(0.5)
    local_path_planner = LocalPathPlanner()
    try:
        while not rospy.is_shutdown():
            local_path_planner.find_optimal_local_path()
    except rospy.ROSInterruptException:
        print("Local path planner node failed")
        pass


if __name__ == '__main__':
    main()
