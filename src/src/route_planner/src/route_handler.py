#!/usr/bin/env python

# ==============================================================================
# --General imports ------------------------------------------------------------
# ==============================================================================
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../perception/EggFiles/carla-0.9.7-py2.7-linux-x86_64.egg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import carla
import time
import carla
import rospy
import numpy as np
import math
from time import sleep
import random
from a_star_algorithm import AStar

# ==============================================================================
# -- ROS imports ---------------------------------------------------------------
# ==============================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from route_planner.msg import RouteLocation
from route_planner.srv import RoutePlanner, RoutePlannerRequest, RoutePlannerResponse
from route_planner.srv import RoutePath, RoutePathRequest, RoutePathResponse
from route_planner.srv import LaneOffsetRoute, LaneOffsetRouteRequest, LaneOffsetRouteResponse
from perception.srv import EgoVehicleName, EgoVehicleNameRequest, EgoVehicleNameResponse
from std_msgs.msg import Bool

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================
DRAW_TIME = 0.1

GLOBAL_DRAW_ROUTE_PATH = False
GLOBAL_DRAW_VEHICLE_ROUTE_POSITION = False
COLORS = {"white": (255, 255, 255), "green": (0, 255, 0), "blue": (30, 144, 255), "yellow": (255, 255, 0),
          "red": (255, 0, 0), "orange": (255, 165, 0), "magenta": (255, 0, 255), "black": (0, 0, 0),
          "lightseagreen": (32, 178, 170), "darkgreen": (0, 100, 0), "darkblue": (0, 10, 240)}


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


class Route:
    def __init__(self):
        self.ego_vehicle_name = "*vehicle.ford.mustang*"
        self.route_path = None
        self.path_index = 0
        self.lateral_offset_from_route = 0.0
        self.route_length = 200
        self.vehicle = None
        self.world = None
        self.map = None
        self.init_world()
        self.global_draw = True
        if os.environ.get('PATHLENGTH') is not None:
            self.route_length = int(os.environ.get('PATHLENGTH'))
        # --- ROS ---
        rospy.init_node('RoutePlanner_node', anonymous=True)

        rospy.Service('route_planner_srv', RoutePlanner, self.handle_route_generation)
        self.client_route_generation()
        rospy.Service('route_path_srv', RoutePath, self.handle_route_path_points)
        rospy.Service('lane_offset_route_srv', LaneOffsetRoute, self.handle_lane_offset_route)
        self.subscriber_global_draw = rospy.Subscriber("global_draw_msg", Bool, self.callback_global_draw, queue_size=1)
        self.pub_localization_in_route_path = rospy.Publisher('localization_in_route_path_msg', RouteLocation, queue_size=1)

    def init_world(self):
        self.client_get_ego_vehicle_name()
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)  # seconds
        world = client.get_world()
        actors_list = world.get_actors()
        self.vehicle = actors_list.filter(self.ego_vehicle_name)[0]
        self.world = self.vehicle.get_world()
        self.map = self.vehicle.get_world().get_map()

    def route_planner(self):
        while not rospy.is_shutdown():
            sleep(0.05)
            self.publish_localization_in_route_path(self.pub_localization_in_route_path)

    def route_generation(self, max_path_length=400, points_dist=1):
        """
        The method represent the route planning system used to generate the route path which the vehicle has to follow.
        The method choose randomly for the available paths to follow one.
        :param max_path_length: The length of the route path
        :param points_dist: The distance between the points of the path
        """
        select = 1
        if select == 1:
            self.route_path = self.generate_random_route_path(max_path_length, points_dist)
            self.route_path = self.get_route_path_points()
        else:
            start = (self.vehicle.get_location().x, self.vehicle.get_location().y)
            end = (random.randint(-300, 300), random.randint(-300, 300))
            self.route_path = []
            route_path = AStar(start, end, self.map, self.world).find_path()
            for i in range(len(route_path)-1):
                yaw = math.degrees(math.atan2(route_path[i+1].y-route_path[i].y, route_path[i+1].x-route_path[i].x))
                self.route_path.append(PathWaypoints(None, x=route_path[i].x, y=route_path[i].y, yaw=yaw))

        if GLOBAL_DRAW_ROUTE_PATH or self.global_draw:
            self.draw_paths(paths=[self.route_path], life_time=1e6, color=COLORS["blue"], same_color=True, symbol="<")
        return self.route_path

    def get_route_path_points(self):
        if self.route_path is None:
            self.route_generation(max_path_length=self.route_length)
        route_path = []
        for w in self.route_path:
            route_path.append(PathWaypoints(w))
        return route_path

    def generate_random_route_path(self, max_path_length=100, points_dist=1):
        """
        Method to find a random path in the world with a specific length given by "max_path_length" variable.
        :param max_path_length: the maximum length of each path
        :param points_dist: the distance of two consecutive waypoints
        :return final_paths: a list of waypoints referred to a driving path
        """
        final_paths = []
        veh_loc = self.vehicle.get_location()
        curr_w = self.map.get_waypoint(veh_loc, lane_type=carla.LaneType.Driving)
        final_paths.append(curr_w)
        path_length = 0.0
        while True:
            next_w = random.choice(curr_w.next(points_dist))
            if next_w is None:
                break
            path_length += next_w.transform.location.distance(curr_w.transform.location)
            if path_length > max_path_length:
                break
            final_paths.append(next_w)
            curr_w = next_w

        return final_paths

    def vehicle_position_in_route_path(self, path_waypoints=None):
        """
        The method is doing a localization of the vehicle into the route path
        :param path_waypoints: A piece of the route path or all the route path if is None
        :return: The point's position of the route path
        """
        if path_waypoints is None:
            path_waypoints = self.route_path
        if self.route_path is None:
            self.path_index = 0
            return self.path_index
        offset = int(len(path_waypoints) / 20)
        offset = 10 if offset < 10 else offset
        p_init = self.path_index - offset
        p_end = self.path_index + offset
        veh_loc = self.vehicle.get_location()
        if len(path_waypoints) < p_end - 1:
            p_end = len(path_waypoints) - 1
        if p_init < 0:
            p_init = 0
        if isinstance(path_waypoints[0], PathWaypoints):
            path_x = [w.x for w in path_waypoints[p_init:p_end]]
            path_y = [w.y for w in path_waypoints[p_init:p_end]]
        else:
            path_x = [w.transform.location.x for w in path_waypoints[p_init:p_end]]
            path_y = [w.transform.location.y for w in path_waypoints[p_init:p_end]]
        x0 = int(self.path_index - p_init)
        x0 = 0 if x0 < 0 else x0
        self.lateral_offset_from_route, rel_index = calc_min_lateral_dist(x_base=path_x, y_base=path_y, x_v=veh_loc.x,
                                                                          y_v=veh_loc.y,
                                                                          x0=x0, offset=offset)
        self.path_index = p_init + rel_index
        # Centering the path in the road
        if isinstance(self.route_path[0], PathWaypoints):
            loc_path = carla.Location(x=self.route_path[self.path_index].x, y=self.route_path[self.path_index].y)
        else:
            loc_path = self.route_path[self.path_index].transform.location
        c_w = self.map.get_waypoint(loc_path, lane_type=carla.LaneType.Driving)
        n_w = c_w.next(0.1)[0]
        if n_w is None or c_w is None:
            return False

        # Find path position relative to road center
        x_mm, y_mm = loc_path.x, loc_path.y
        Ax, Ay = c_w.transform.location.x,  c_w.transform.location.y
        Bx, By = n_w.transform.location.x,  n_w.transform.location.y
        curve_side = np.sign((Bx - Ax) * (y_mm - Ay) - (By - Ay) * (x_mm - Ax))
        curve_side = 1.0 if curve_side == 0.0 else curve_side
        path_dist_from_road_center = curve_side*math.hypot(c_w.transform.location.y-y_mm, c_w.transform.location.x-x_mm)

        if self.path_index + 1 < len(self.route_path) - 1:
            x_mm, y_mm = self.vehicle.get_location().x, self.vehicle.get_location().y
            Ax = self.route_path[self.path_index].transform.location.x if not isinstance(self.route_path[0], PathWaypoints) else self.route_path[self.path_index].x
            Ay = self.route_path[self.path_index].transform.location.y if not isinstance(self.route_path[0], PathWaypoints) else self.route_path[self.path_index].y
            Bx = self.route_path[self.path_index + 1].transform.location.x if not isinstance(self.route_path[0], PathWaypoints) else self.route_path[self.path_index+1].x
            By = self.route_path[self.path_index + 1].transform.location.y if not isinstance(self.route_path[0], PathWaypoints) else self.route_path[self.path_index+1].y
            curve_side = np.sign((Bx - Ax) * (y_mm - Ay) - (By - Ay) * (x_mm - Ax))
            curve_side = 1.0 if curve_side == 0.0 else curve_side
            self.lateral_offset_from_route = curve_side * self.lateral_offset_from_route + path_dist_from_road_center

        if GLOBAL_DRAW_VEHICLE_ROUTE_POSITION or self.global_draw:
            if isinstance(self.route_path[0], PathWaypoints):
                loc_v = carla.Location(x=self.route_path[self.path_index].x, y=self.route_path[self.path_index].y)
            else:
                loc_v = self.route_path[self.path_index].transform.location
            self.world.debug.draw_string(loc_v, " <---- Vehicle in route ", draw_shadow=False,
                                         color=carla.Color(r=255, g=220, b=0), life_time=0.1, persistent_lines=True)
        return self.path_index >= len(self.route_path) - 3

    def get_lane_offset_and_centering(self, lane_id=-1, current_route_offset=0.0):
        """
        The method give the lateral offset from the route path for the left lanes for lane_id < 0 or for the right lanes
        for lane_id > 0. The number indicate the lane id number. For example if lane_id=-1, -2, -3 ect indicates the first
        left lane, the second left lane, the third left lane ect
        :param lane_id: The id number for the lane is taken, a negative sign gives the left lanes,a positive the right ones.
        :param current_route_offset: The ego vehicle's distance from the route
        :return: The lateral offset from the route for the desired lane
        """
        current_w = self.map.get_waypoint(self.vehicle.get_location(), lane_type=carla.LaneType.Driving)
        waypoint = current_w
        current_yaw = self.vehicle.get_transform().rotation.yaw

        if lane_id < 0.0:
            for i in range(abs(lane_id)):
                if 90.0 < (current_yaw - waypoint.transform.rotation.yaw) % 360.0 < 270.0:
                    w = waypoint.get_right_lane()
                else:
                    w = waypoint.get_left_lane()
                if w is not None:
                    if w.lane_type in [carla.LaneType.Driving, carla.LaneType.Bidirectional, carla.LaneType.Parking]:
                        waypoint = w
        elif lane_id > 0.0:
            for i in range(abs(lane_id)):
                if 90.0 < (current_yaw - waypoint.transform.rotation.yaw) % 360.0 < 270.0:
                    w = waypoint.get_left_lane()
                else:
                    w = waypoint.get_right_lane()
                if w is not None:
                    if w.lane_type in [carla.LaneType.Driving, carla.LaneType.Bidirectional, carla.LaneType.Parking]:
                        waypoint = w
        distance = current_w.transform.location.distance(waypoint.transform.location)
        start_d1 = abs(current_route_offset) - (current_w.lane_width / 2)
        start_d1 = 0.0 if start_d1 < 0.0 else start_d1
        current_lane_id = math.ceil(start_d1 / current_w.lane_width)
        sign_lane = -1.0 if current_route_offset < 0.0 else 1.0
        sign_offset = -1.0 if lane_id < 0.0 else 1.0
        distance = sign_offset * distance + sign_lane * current_lane_id * current_w.lane_width
        return distance

    def draw_paths(self, paths, life_time, color=None, same_color=True, symbol="O"):
        """
       Method to draw the paths in simulator spectator
       :param same_color: If true the paths have same color
       :param paths :A list of lists of waypoints that represent paths or a list of list of Locations
       :param life_time: The time the waypoints are drawn
       :param color: A list with RGB values of the path color
       :param symbol: The symbol of the draw point
        """
        if color is None:
            color = [225, 0, 0]
        if not paths[0]:  # <-- There is no path
            return 0
        if type(paths[0][0]) is carla.Waypoint:
            for path in paths:
                id_path = paths.index(path)
                r = (id_path + 2) ** 2 % 225
                g = ((id_path + 12) ** 2 + 13) % 225
                b = ((id_path + 23) ** 2) % 255
                for w in path:
                    if same_color:
                        self.world.debug.draw_string(w.transform.location, symbol, draw_shadow=False,
                                                     color=carla.Color(r=color[0], g=color[1], b=color[2]),
                                                     life_time=life_time,
                                                     persistent_lines=True)
                        sleep(0.00000000000000001)

                    else:
                        self.world.debug.draw_string(w.transform.location, symbol, draw_shadow=False,
                                                     color=carla.Color(r=r, g=g, b=b), life_time=life_time,
                                                     persistent_lines=True)
                        sleep(0.00000000000000001)
        elif type(paths[0][0]) is carla.Location:
            for path in paths:
                id_path = paths.index(path)
                r = (id_path + 2) ** 2 % 225
                g = ((id_path + 12) ** 2 + 13) % 225
                b = ((id_path + 23) ** 2) % 255
                for location in path:
                    if same_color:
                        self.world.debug.draw_string(location, symbol, draw_shadow=False,
                                                     color=carla.Color(r=color[0], g=color[1], b=color[2]),
                                                     life_time=life_time,
                                                     persistent_lines=True)
                        sleep(0.00000000000000001)

                    else:
                        self.world.debug.draw_string(location, symbol, draw_shadow=False,
                                                     color=carla.Color(r=r, g=g, b=b), life_time=life_time,
                                                     persistent_lines=True)
                        sleep(0.00000000000000001)
        else:
            for id_path, path in enumerate(paths):
                r = (id_path + 2) ** 2 % 225
                g = ((id_path + 12) ** 2 + 13) % 225
                b = ((id_path + 23) ** 2) % 255
                for w in path:
                    if same_color:
                        self.world.debug.draw_string(carla.Location(x=w.x, y=w.y), symbol, draw_shadow=False,
                                                     color=carla.Color(r=color[0], g=color[1], b=color[2]),
                                                     life_time=life_time,
                                                     persistent_lines=True)
                        sleep(0.00000000000000001)

                    else:
                        self.world.debug.draw_string(carla.Location(x=w.x, y=w.y), symbol, draw_shadow=False,
                                                     color=carla.Color(r=r, g=g, b=b), life_time=life_time,
                                                     persistent_lines=True)
                        sleep(0.00000000000000001)

    def publish_localization_in_route_path(self, pub):
        end_of_path = RouteLocation()
        end_of_path.end_of_path = self.vehicle_position_in_route_path()
        end_of_path.route_index = self.path_index
        end_of_path.lateral_offset = self.lateral_offset_from_route
        #rospy.loginfo(end_of_path)
        pub.publish(end_of_path)

    def handle_lane_offset_route(self, request):
        lane_offset_route = self.get_lane_offset_and_centering(request.lane_id, self.lateral_offset_from_route)
        return lane_offset_route

    def handle_route_generation(self, request):
        route_path = self.route_generation(max_path_length=request.path_length)
        return 1 if route_path is not None else 0

    def handle_route_path_points(self, request):
        response = RoutePathResponse()
        route_path = self.route_path
        response.route_path = route_path
        return response

    def callback_global_draw(self, ros_data):
        self.global_draw = ros_data.data

    # Services clients
    def client_route_generation(self):
        rospy.wait_for_service('route_planner_srv')
        try:
            route_planner = rospy.ServiceProxy('route_planner_srv', RoutePlanner)
            resp1 = route_planner(self.route_length)
            #rospy.loginfo(resp1)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def client_get_ego_vehicle_name(self):
        try:
            rospy.wait_for_service('ego_vehicle_name_srv', timeout=4.0)
            ego_vehicle_name = rospy.ServiceProxy('ego_vehicle_name_srv', EgoVehicleName)
            resp1 = ego_vehicle_name(0)
            self.ego_vehicle_name = resp1.name
            #rospy.loginfo(resp1)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

if __name__ == '__main__':
    route = Route()
    try:
        route.route_planner()
    except rospy.ROSInterruptException:
        pass





