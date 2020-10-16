#!/usr/bin/env python

# ==============================================================================
# --General imports ------------------------------------------------------------
# ==============================================================================
import time
import math
import rospy
import os
import sys
from time import sleep
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../perception/EggFiles/carla-0.9.7-py2.7-linux-x86_64.egg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import carla
import collections
import weakref
import psutil

# ==============================================================================
# -- ROS imports ---------------------------------------------------------------
# ==============================================================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from perception.msg import Object
from perception.msg import TrafficSigns
from perception.msg import WaypointsList
from perception.msg import Waypoint
from perception.msg import ObjectsList
from perception.msg import Lane
from perception.msg import RoadLanes
from perception.msg import TrajectoriesList
from perception.msg import Float64List
from perception.msg import StringList
from control.msg import VehicleCmd
from local_path_planner.msg import LocalOptimalPath
from std_msgs.msg import Float64
from std_msgs.msg import Bool

from perception.srv import EgoVehicleGeometry, EgoVehicleGeometryRequest, EgoVehicleGeometryResponse
from perception.srv import VehicleByID, VehicleByIDRequest, VehicleByIDResponse
from perception.srv import DrivingPaths, DrivingPathsRequest, DrivingPathsResponse
from perception.srv import EgoVehicleName, EgoVehicleNameRequest, EgoVehicleNameResponse
from route_planner.srv import RoutePath, RoutePathRequest, RoutePathResponse
from route_planner.msg import RouteLocation
from maneuver_generator.msg import ManeuverDataROS


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


class TrafficSigns:
    """
    The class represent a driving traffic signs with the information needed
    """

    def __init__(self):
        self.traffic_light_exist = False
        self.traffic_light_distance = -1.0
        self.traffic_light_state = "None"
        self.stop_sign_exist = False
        self.stop_sign_distance = -1.0
        self.speed_sign_exist = False
        self.speed_sign_distance = -1.0
        self.traffic_junction_exist = False
        self.traffic_junction_distance = -1.0
        self.speed_limit = -1.0


class PathWaypoints:
    def __init__(self, waypoint, x=0.0, y=0.0, yaw=0.0):
        """
        :param vehicle: the vehicle to find local map
        :param visibility_radius: the radius around the vehicle that the map is been created in
        """
        if waypoint is not None:
            self.x = waypoint.transform.location.x
            self.y = waypoint.transform.location.y
            self.yaw = waypoint.transform.rotation.yaw
        else:
            self.x = x
            self.y = y
            self.yaw = yaw


class CollisionSensor(object):
    def __init__(self, parent_actor):
        # Collect data
        self.pedestrians_collisions_num = 0
        self.vehicles_collisions_num = 0
        self.statics_collisions_num = 0
        self.pedestrians_collisions_intensity = []
        self.vehicles_collisions_intensity = []
        self.statics_collisions_intensity = []
        self.sensor = None
        self._parent = parent_actor
        self.prev_time = time.time()
        self.time_interval = 2
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self or time.time()-self.prev_time < self.time_interval:
            return
        self.prev_time = time.time()
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        if "pedestrian" in event.other_actor.type_id.split("."):
            self.pedestrians_collisions_num += 1
            self.pedestrians_collisions_intensity.append(intensity)
        elif "vehicle" in event.other_actor.type_id.split("."):
            self.vehicles_collisions_num += 1
            self.vehicles_collisions_intensity.append(intensity)
        else:
            self.statics_collisions_num += 1
            self.statics_collisions_intensity.append(intensity)


class Evaluation:
    def __init__(self):
        self.file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_info_save.txt")
        #with open(self.file_path, "a+") as file:
            #file.truncate()
        self.ego_vehicle_name = "*vehicle.ford.mustang*"
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)  # seconds
        self.world = client.get_world()
        actors_list = self.world.get_actors()
        self.vehicle = actors_list.filter(self.ego_vehicle_name)[0]
        self.map = self.vehicle.get_world().get_map()
        self.ego_vehicle = None
        # Route
        self.route_path = []
        self.path_index = 0
        # Collision
        self.collision_sensor = CollisionSensor(self.vehicle)
        # Traffic light
        self.traffic_signs = TrafficSigns()
        self.prev_time = time.time()
        self.wait_time = 2
        self.stop_sign_checked = False
        self.sign_ok = False
        self.violation_detected = False
        self.stop_violation_num = 0
        self.light_violation_num = 0
        self.speed_limit_violation = 0
        # Off road event
        self.start_time = time.time()
        self.prev_off_road_time = time.time()
        self.off_road_time = 0
        # Off lane event
        self.prev_off_lane_time = time.time()
        self.off_lane_time = 0
        # Block
        self.block_time = time.time()
        self.block_time_wait = 20
        # Speed limit
        self.prev_time_speed_limit = time.time()
        self.speed_limit = 30
        self.prev_speed_limit = self.speed_limit
        self.speed_limit_delay = 3
        # --- ROS ---
        rospy.init_node('evaluation_node', anonymous=True)
        self.client_get_route_path()
        self.subscriber_ego_vehicle = rospy.Subscriber("ego_vehicle_msg", Object, self.callback_ego_vehicle,
                                                       queue_size=1)
        sleep(2)
        self.pre_loc = [self.ego_vehicle.x, self.ego_vehicle.y]

    def route_completion(self):
        """
        The method is doing a localization of the vehicle into the route path
        :return: The point's position of the route path
        """
        if self.ego_vehicle is None:
            return 0
        path_waypoints = self.route_path
        offset = int(len(path_waypoints) / 20)
        offset = 10 if offset < 10 else offset
        p_init = self.path_index - offset
        p_end = self.path_index + offset
        #veh_loc = self.vehicle.get_location()
        if len(path_waypoints) < p_end - 1:
            p_end = len(path_waypoints) - 1
        if p_init < 0:
            p_init = 0
        if isinstance(path_waypoints[0], PathWaypoints):
            path_x = [w.x for w in path_waypoints[p_init:p_end]]
            path_y = [w.y for w in path_waypoints[p_init:p_end]]
        else:
            path_x = [w.x for w in path_waypoints[p_init:p_end]]
            path_y = [w.y for w in path_waypoints[p_init:p_end]]
        x0 = int(self.path_index - p_init)
        x0 = 0 if x0 < 0 else x0
        _, rel_index = calc_min_lateral_dist(x_base=path_x, y_base=path_y, x_v=self.ego_vehicle.x,
                                                                          y_v=self.ego_vehicle.y,
                                                                          x0=x0, offset=offset)
        self.path_index = p_init + rel_index
        complete_percentage = round(float(self.path_index)/float(len(self.route_path)), 2)
        complete_percentage = 1 if complete_percentage > 0.97 else complete_percentage
        return complete_percentage

    def get_collisions(self):
        return self.collision_sensor.vehicles_collisions_num, self.collision_sensor.pedestrians_collisions_num, self.collision_sensor.statics_collisions_num

    def check_sign_violation(self):
        if self.ego_vehicle is None:
            return
        stop_violation = False
        light_violation = False
        self.get_traffic_signs()
        traffic_light_exist = self.traffic_signs.traffic_light_exist and self.traffic_signs.traffic_light_state != "Green"
        if self.traffic_signs.stop_sign_exist:
            self.stop_sign_checked = True

        if self.traffic_signs.stop_sign_exist and self.ego_vehicle.speed < 0.2 and not self.sign_ok:
            self.sign_ok = True

        if not self.traffic_signs.stop_sign_exist and self.stop_sign_checked and not self.sign_ok:
            stop_violation = True
        if traffic_light_exist and abs(self.traffic_signs.traffic_light_distance) < 0.1:
            light_violation = True

        if stop_violation and not self.violation_detected:
            self.stop_violation_num += 1
            self.violation_detected = True
        if light_violation and not self.violation_detected:
            self.light_violation_num += 1
            self.violation_detected = True

        if not self.traffic_signs.stop_sign_exist:
            self.stop_sign_checked = False
            self.sign_ok = False
        if not self.traffic_signs.stop_sign_exist and not traffic_light_exist:
            self.violation_detected = False

        return self.stop_violation_num, self.light_violation_num

    def check_speed_limit_violation(self):
        if self.traffic_signs.speed_sign_exist:
            self.speed_limit = self.traffic_signs.speed_limit
        if self.prev_speed_limit != self.speed_limit:
            self.prev_time_speed_limit == time.time()
            self.prev_speed_limit = self.speed_limit
            return self.speed_limit_violation
        if time.time() - self.prev_speed_limit < self.speed_limit_delay:
            return self.speed_limit_violation
        if self.speed_limit + 5 < self.ego_vehicle.speed * 3.6 and time.time() - self.prev_time > self.wait_time:
            self.speed_limit_violation += 1
            self.prev_time = time.time()
        return self.speed_limit_violation

    def off_road_event(self):
        w_n = self.map.get_waypoint(self.vehicle.get_location(), lane_type=carla.LaneType.Any)
        if not str(w_n.lane_type) in ["Driving", "Bidirectional", "Shoulder", "Parking"]:
            self.off_road_time += time.time() - self.prev_off_road_time
        self.prev_off_road_time = time.time()
        return self.off_road_time

    def off_lane_event(self):
        w_n = self.map.get_waypoint(self.vehicle.get_location(), lane_type=carla.LaneType.Any)
        if not str(w_n.lane_type) in ["Driving", "Bidirectional", "Shoulder", "Parking"]:
            self.off_lane_time += time.time() - self.prev_off_lane_time
        self.prev_off_lane_time = time.time()
        return self.off_lane_time

    def check_vehicle_got_blocked(self):
        dist = math.hypot(self.pre_loc[0]-self.ego_vehicle.x, self.pre_loc[1]-self.ego_vehicle.y)
        if dist > 0.1:
            self.block_time = time.time()
        elif time.time()-self.block_time > self.block_time_wait:
            return True
        if self.collision_sensor.vehicles_collisions_num + self.collision_sensor.pedestrians_collisions_num \
                + self.collision_sensor.statics_collisions_num > 30:
            return True
        return False

    def check_run_time_expired(self):
        minute10 = 10*60
        if time.time()-self.start_time > minute10:
            return True
        return False

    def get_traffic_signs(self, dist=20.0):
        """
        Method to detect traffic signs at dist_ahead distance from the vehicle. If there is a route path then it use it
        to predict the direction of the vehicle, otherwise the road geometry is used to take the path of the road
        direction. In signs list are contained the traffic lights, stops and speed limit signs only.
        :param dist: The maximum distance ahead the vehicle in which it can detect traffic signs
        :return: Boolean values of each case if any sign exists, with the distances from the vehicle of nearest traffic
        light, stop sign and speed limit sign with the speed limit value.
        """
        max_dist = 3.0
        norm_factor = 2.5
        # Get waypoint front of vehicle
        curr_w = self.route_path[self.path_index]
        step = 2.0
        dist_sum = 0.0
        w_list = [curr_w]
        curvature_list = [0.0]
        curvature_sum = 0.0
        point_id = self.path_index
        while dist_sum < dist:
            curr_w = w_list[-1]
            point_id += 1
            if point_id < len(self.route_path) - 1:
                next_w = self.route_path[point_id]
            else:
                break
            relative_distance = math.hypot(next_w.y-curr_w.y, next_w.x-curr_w.x) + 0.001
            dist_sum += relative_distance
            w_list.append(next_w)
            theta1 = curr_w.yaw
            if theta1 < 0.0:
                theta1 = 360.0 + theta1
            if theta1 > 180.0:
                theta1 = theta1 - 360.0
            theta2 = next_w.yaw
            if theta2 < 0.0:
                theta2 = 360.0 + theta2
            if theta2 > 180.0:
                theta2 = theta2 - 360.0
            angle_diff = theta2 - theta1
            if angle_diff > 180.0:
                angle_diff = 360.0 - angle_diff
            elif angle_diff < -180.0:
                angle_diff = -360.0 - angle_diff
            curvature = abs(angle_diff / relative_distance)
            curvature_sum += curvature
            curvature_list.append(curvature)

        # If there is a sharp increase in the curvature of the path we assume that we cannot use the point from there
        # and after due to a coming turn
        index = len(curvature_list) - 1
        curvature_average = norm_factor * curvature_sum / (index + 1)
        for c in curvature_list[7:]:
            if c > curvature_average and abs(c - curvature_average) > 0.1 and c > 2.0:
                index = curvature_list.index(c)
                break
        w_list = w_list[:index]
        # We will use the waypoints found before to search for traffic signs
        # Get traffic lights
        traffic_signs = self.world.get_actors().filter('*traffic.traffic_light*')
        trigger_points_loc = []
        step_ahead = 6.0
        for traffic_sign in traffic_signs:
            trigger = traffic_sign.trigger_volume
            traffic_sign.get_transform().transform(trigger.location)
            sign_w = self.map.get_waypoint(trigger.location, lane_type=carla.LaneType.Driving)
            theta = math.radians(sign_w.transform.rotation.yaw)
            b_x = trigger.location.x + step_ahead * math.cos(theta)
            b_y = trigger.location.y + step_ahead * math.sin(theta)
            trigger.location.x = b_x
            trigger.location.y = b_y
            trigger_points_loc.append(trigger.location)

        flag_break = False
        light_distance = -1.0
        traffic_light_exist = False
        traffic_light_state = "None"
        for w in w_list:
            for trigger_loc in trigger_points_loc:
                distance_to_path = math.hypot(w.y-trigger_loc.y, w.x-trigger_loc.x)
                if distance_to_path <= max_dist:
                    traffic_light_exist = True
                    light_distance = trigger_loc.distance(self.vehicle.get_location())
                    traffic_light_state = str(traffic_signs[trigger_points_loc.index(trigger_loc)].get_state())
                    flag_break = True
                    break
            if flag_break:
                break

        # Get traffic stop signs
        traffic_signs = self.world.get_actors().filter('*traffic.stop*')
        trigger_points_loc = []
        step_ahead = 2.0
        for traffic_sign in traffic_signs:
            trigger = traffic_sign.trigger_volume
            traffic_sign.get_transform().transform(trigger.location)
            sign_w = self.map.get_waypoint(trigger.location, lane_type=carla.LaneType.Driving)
            theta = math.radians(sign_w.transform.rotation.yaw)
            b_x = trigger.location.x + step_ahead * math.cos(theta)
            b_y = trigger.location.y + step_ahead * math.sin(theta)
            trigger.location.x = b_x
            trigger.location.y = b_y
            trigger_points_loc.append(trigger.location)

        flag_break = False
        stop_sign_distance = -1.0
        stop_sign_flag = False
        for w in w_list[1:]:
            for trigger_loc in trigger_points_loc:
                distance_to_path = math.hypot(w.y-trigger_loc.y, w.x-trigger_loc.x)
                if distance_to_path <= max_dist:
                    stop_sign_flag = True
                    stop_sign_distance = trigger_loc.distance(self.vehicle.get_location())
                    flag_break = True
                    break
            if flag_break:
                break

        # Get traffic speed limit signs
        traffic_signs = self.world.get_actors().filter('*traffic.speed_limit*')
        trigger_points_loc = []
        step_ahead = 1.0
        for traffic_sign in traffic_signs:
            trigger = traffic_sign.trigger_volume
            traffic_sign.get_transform().transform(trigger.location)
            sign_w = self.map.get_waypoint(trigger.location, lane_type=carla.LaneType.Driving)
            theta = math.radians(sign_w.transform.rotation.yaw)
            b_x = trigger.location.x + step_ahead * math.cos(theta)
            b_y = trigger.location.y + step_ahead * math.sin(theta)
            trigger.location.x = b_x
            trigger.location.y = b_y
            trigger_points_loc.append(trigger.location)

        flag_break = False
        speed_sign_distance = -1.0
        speed_sign_flag = False
        speed_limit_value = -1.0
        for w in w_list[3:]:
            for trigger_loc in trigger_points_loc:
                distance_to_path = math.hypot(w.y-trigger_loc.y, w.x-trigger_loc.x)
                if distance_to_path <= max_dist:
                    speed_sign_flag = True
                    speed_sign_distance = trigger_loc.distance(self.vehicle.get_location())
                    speed_limit_value = [int(s) for s in str(traffic_signs[trigger_points_loc.index(trigger_loc)].type_id).split(".") if s.isdigit()][0]
                    flag_break = True
                    break
            if flag_break:
                break

        # Get traffic junction
        dist_from_junction = -1.0
        there_is_junction = False
        for w in w_list:
            w_location = carla.Location(x=w.x, y=w.y)
            is_junction = self.map.get_waypoint(w_location, lane_type=carla.LaneType.Driving).is_junction
            if is_junction:
                there_is_junction = True
                loc_veh = self.vehicle.get_location()
                dist_from_junction = math.hypot(w.y-loc_veh.y, w.x-loc_veh.x)
                dist_from_junction = 0.0 if w is w_list[0] else dist_from_junction
                break

        self.traffic_signs.traffic_light_exist = traffic_light_exist
        self.traffic_signs.traffic_light_distance = light_distance
        self.traffic_signs.traffic_light_state = traffic_light_state
        self.traffic_signs.stop_sign_exist = stop_sign_flag
        self.traffic_signs.stop_sign_distance = stop_sign_distance
        self.traffic_signs.speed_sign_exist = speed_sign_flag
        self.traffic_signs.speed_sign_distance = speed_sign_distance
        self.traffic_signs.traffic_junction_exist = there_is_junction
        self.traffic_signs.traffic_junction_distance = dist_from_junction
        self.traffic_signs.speed_limit = speed_limit_value

        return self.traffic_signs

    # ------ ROS -------
    def client_get_route_path(self):
        rospy.wait_for_service('route_path_srv')
        try:
            route_path_srv = rospy.ServiceProxy('route_path_srv', RoutePath)
            resp1 = route_path_srv(0)
            #rospy.loginfo(resp1)
            self.route_path = resp1.route_path
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def callback_ego_vehicle(self, ros_data):
        self.ego_vehicle = ros_data
        # rospy.loginfo(ros_data)

    def write_in_file(self):

        with open(self.file_path, "a+") as file:
            if os.stat(self.file_path).st_size == 0:
                file.write("Route_completion,")
                file.write("Vehicles_collisions_number,")
                file.write("Pedestrians_collisions_number,")
                file.write("Statics_collisions_number,")
                file.write("Stop_violations_number,")
                file.write("Light_violations_number,")
                file.write("Speed_limit_violations_number,")
                file.write("Vehicle_got_blocked,")
                file.write("Off_road_event_time")

                file.write("\n")

            file.write(str(self.route_completion()) + ",")
            v_n, p_n, s_n = self.get_collisions()
            file.write(str(v_n) + ",")
            file.write(str(p_n) + ",")
            file.write(str(s_n) + ",")
            self.check_sign_violation()
            file.write(str(self.stop_violation_num) + ",")
            file.write(str(self.light_violation_num) + ",")
            file.write(str(self.check_speed_limit_violation()) + ",")
            if evaluation.check_vehicle_got_blocked():
                file.write("1,")
            else:
                file.write("0,")
            time_percentage = self.off_road_event() / (time.time() - self.start_time)
            file.write(str(round(1 - time_percentage, 2)))
            file.write("\n")


if __name__ == '__main__':
    evaluation = Evaluation()
    try:
        while not rospy.is_shutdown():
            evaluation.check_speed_limit_violation()
            evaluation.check_sign_violation()
            evaluation.off_road_event()

            if evaluation.route_completion() == 1 or evaluation.check_vehicle_got_blocked() or evaluation.check_run_time_expired():
                evaluation.write_in_file()
                sys.exit()

    except rospy.ROSInterruptException:
        print("Local path planner node failed")
        pass
