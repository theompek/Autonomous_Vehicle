#!/usr/bin/env python

"""
This module implements a module with necessary methods which gives information
for the local environment around the vehicle.
Module functions:
1) Available driving waypoints: A list of points of the local map which the vehicle can drive in
2) Objects around the car: The position and orientation of objects like other cars and pedestrians relative to our vehicle
3) Road limits: The length and the limits of the driving road, the sidewalks etc


"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../EggFiles/carla-0.9.7-py2.7-linux-x86_64.egg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import carla
import time
from time import sleep
import math
import numpy as np
import random
import copy
from objects import Objects
from borders import Border


# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================
DRAW_TIME = 0.1
GLOBAL_DRAW = True
GLOBAL_DRAW_TRAFFIC = False
GLOBAL_DRAW_EGO_TRAJECTORY = False
GLOBAL_DRAW_LANES_VEHICLES_ARROWS = False
GLOBAL_DRAW_LANES_AVAILABILITY = False
GLOBAL_DRAW_LANES_REPRESENTATION = False
COLORS = {"white": (255, 255, 255), "green": (0, 255, 0), "blue": (30, 144, 255), "yellow": (255, 255, 0),
          "red": (255, 0, 0), "orange": (255, 165, 0), "magenta": (255, 0, 255), "black": (0, 0, 0),
          "lightseagreen": (32, 178, 170), "darkgreen": (0, 100, 0), "darkblue": (0, 10, 240)}


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


class Lane:
    """
    The class represent a driving lane with the information needed
    """

    def __init__(self):
        self.availability = False  # There is any space for driving
        self.lane_type = "Driving"
        self.front_front_vehicle = None
        self.front_vehicle = None
        self.rear_vehicle = None
        self.rear_rear_vehicle = None
        self.vehicle_num_front = 0
        self.vehicle_num_rear = 0
        self.lateral_offset_front_front = 0.0
        self.lateral_offset_front = 0.0
        self.lateral_offset_rear = 0.0
        self.lateral_offset_front_rear = 0.0
        self.lane_width = 0.0
        self.opposite_direction = True  # If False then vehicle can step on lane for a short duration

    def unavailable_lane(self):
        self.availability = False
        self.lane_type = "Driving"
        self.front_front_vehicle = None
        self.front_vehicle = None
        self.rear_vehicle = None
        self.rear_rear_vehicle = None
        self.vehicle_num_front = 0
        self.vehicle_num_rear = 0
        self.lateral_offset_front_front = 0.0
        self.lateral_offset_front = 0.0
        self.lateral_offset_rear = 0.0
        self.lateral_offset_front_rear = 0.0
        self.lane_width = 0.0
        self.opposite_direction = True


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


class LocalMap:
    """
        The class includes all the functions to retrieve information about the vehicle's surround environment,
        for example available paths for the vehicle to follow, obstacles in the road like other vehicles and pedestrians,
        general information related to local environment map.
    """

    def __init__(self, vehicle, visibility_radius=70.0):
        """
        :param vehicle: the vehicle to find local map
        :param visibility_radius: the radius around the vehicle that the map is been created in
        """
        self.vehicle = vehicle
        self.world = self.vehicle.get_world()
        self.map = self.vehicle.get_world().get_map()
        self.visibility_radius = visibility_radius
        self.route_path = None
        self.path_index = 0
        self.current_Lane = Lane()
        self.left_Lane = Lane()
        self.right_Lane = Lane()
        self.right_waypoint = None
        self.left_waypoint = None
        self.traffic_signs = TrafficSigns()
        self.p_d_add_noise = False  # Position and direction noise
        self.p_d_noise_mean = 0.0
        self.p_d_noise_std = 0.0
        self.s_add_noise = False  # Speed noise
        self.s_noise_mean = 0.0
        self.s_noise_std = 0.0

    def apply_vehicle_control(self, throttle, steer_angle, brake):
        veh_control = self.vehicle.get_control()
        veh_control.throttle = throttle
        veh_control.steer = steer_angle
        veh_control.brake = brake
        self.vehicle.apply_control(veh_control)

    def set_route_path(self, route_path):
        self.route_path = []
        for point in route_path:
            waypoint = PathWaypoints(None, x=point.x, y=point.y, yaw=point.yaw)
            self.route_path.append(waypoint)

    def get_simulator_time(self):
        return self.world.get_snapshot().timestamp.elapsed_seconds

    def get_ego_vehicle(self):
        return Objects("ego_vehicle", self.vehicle)

    def add_noise_in_objects_position_direction_and_speed(self, p_d_add_noise=False, p_d_std=0, s_add_noise=False, s_std=0):
        self.p_d_add_noise = p_d_add_noise  # Position and direction noise
        self.p_d_noise_std = p_d_std
        self.s_add_noise = s_add_noise  # Speed noise
        self.s_noise_std = s_std

    def get_vehicle_by_id(self, vehicle_id):
        overtaking_vehicle = self.world.get_actor(vehicle_id)
        if overtaking_vehicle is None:
            return None
        return Objects("vehicle", overtaking_vehicle)

    def get_ego_vehicle_length_and_max_steer_angle(self):
        # Calculate distance between front and rear wheels
        i, j = 0, 2
        loc1 = carla.Location()
        loc1.x = self.vehicle.get_physics_control().wheels[i].position.x
        loc1.y = self.vehicle.get_physics_control().wheels[i].position.y
        loc1.z = self.vehicle.get_physics_control().wheels[i].position.z

        loc2 = carla.Location()
        loc2.x = self.vehicle.get_physics_control().wheels[j].position.x
        loc2.y = self.vehicle.get_physics_control().wheels[j].position.y
        loc2.z = self.vehicle.get_physics_control().wheels[j].position.z
        L = loc1.distance(loc2) / 100
        # Get vehicle max steering angle
        physics_control = self.vehicle.get_physics_control()
        max_angle = physics_control.wheels[0].max_steer_angle
        return L, max_angle

    def get_traffic_signs(self, dist=50.0):
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

        if GLOBAL_DRAW or GLOBAL_DRAW_TRAFFIC and False:
            [self.draw_text(carla.Location(x=w.x, y=w.y), " C ", COLORS["lightseagreen"], life_time=DRAW_TIME) for w in w_list]
        # If there is a sharp increase in the curvature of the path we assume that we cannot use the point from there
        # and after due to a coming turn
        index = len(curvature_list) - 1
        curvature_average = norm_factor * curvature_sum / (index + 1)
        for c in curvature_list[7:]:
            if c > curvature_average and abs(c - curvature_average) > 0.1 and c > 2.0:
                index = curvature_list.index(c)
                break
        w_list = w_list[:index]
        if GLOBAL_DRAW or GLOBAL_DRAW_TRAFFIC:
            [self.draw_text(carla.Location(x=w.x, y=w.y), "    * ", COLORS["darkgreen"], life_time=DRAW_TIME) for w in w_list]

        # We will use the waypoints found before to search for traffic signs
        # Get traffic lights
        traffic_signs = self.world.get_actors().filter('*traffic.traffic_light*')
        trigger_points_loc = []
        step_ahead = 0.1
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
                    if GLOBAL_DRAW or GLOBAL_DRAW_TRAFFIC:
                        tr_text = "<----- "+str(traffic_signs[trigger_points_loc.index(trigger_loc)].type_id) + \
                                  "  " + traffic_light_state
                        self.world.debug.draw_string(trigger_loc, tr_text, draw_shadow=False,
                                                     color=carla.Color(r=255, g=20, b=0), life_time=DRAW_TIME, persistent_lines=True)
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
                    if GLOBAL_DRAW or GLOBAL_DRAW_TRAFFIC:
                        tr_text = "<---- " + str(traffic_signs[trigger_points_loc.index(trigger_loc)].type_id)
                        self.world.debug.draw_string(trigger_loc, tr_text, draw_shadow=False, color=carla.Color(r=255, g=20, b=0),
                                                     life_time=DRAW_TIME, persistent_lines=True)
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
                    speed_limit_value = \
                    [int(s) for s in str(traffic_signs[trigger_points_loc.index(trigger_loc)].type_id).split(".") if
                     s.isdigit()][0]
                    if GLOBAL_DRAW or GLOBAL_DRAW_TRAFFIC:
                        tr_text = "<---- " + str(traffic_signs[trigger_points_loc.index(trigger_loc)].type_id)
                        self.world.debug.draw_string(trigger_loc, tr_text, draw_shadow=False, color=carla.Color(r=255, g=20, b=0),
                                                     life_time=DRAW_TIME, persistent_lines=True)
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
                if GLOBAL_DRAW or GLOBAL_DRAW_TRAFFIC:
                    self.world.debug.draw_string(w_location, " > * <       Is junction", draw_shadow=False,
                                                 color=carla.Color(r=2, g=200, b=0), life_time=DRAW_TIME, persistent_lines=True)
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

    def get_ego_vehicle_possible_trajectory(self, path_length=50, interval_dist=2):
        index = self.path_index
        path = [self.route_path[index]]
        dist_temp = 0.0
        while True:
            index += 1
            if index + 1 > len(self.route_path) - 1:
                break
            w = self.route_path[index]
            next_w = self.route_path[index+1]
            dist_temp += math.hypot(next_w.y - w.y, next_w.x - w.x)
            if dist_temp > path_length:
                break
            points_dist = math.hypot(path[-1].y - w.y, path[-1].x - w.x)
            if points_dist > interval_dist:
                path.append(w)

        trajectory = [PathWaypoints(None, x=self.vehicle.get_location().x, y=self.vehicle.get_location().y,
                                    yaw=path[0].yaw)]
        for index in range(len(path) - 1):
            dist = math.hypot(path[index+1].y - path[index].y, path[index+1].x - path[index].x)
            theta = math.radians(path[index].yaw)
            b_x = trajectory[-1].x + dist * math.cos(theta)
            b_y = trajectory[-1].y + dist * math.sin(theta)
            trajectory.append(PathWaypoints(None, x=b_x, y=b_y, yaw=math.degrees(theta)))

        if GLOBAL_DRAW_EGO_TRAJECTORY or GLOBAL_DRAW:
            self.draw_paths([trajectory], life_time=DRAW_TIME, color=COLORS["magenta"], same_color=True, symbol="^")
        return trajectory

    def get_traffic_stop_and_lights_objects_location(self):
        curr_location, max_dist = self.vehicle.get_location(), self.visibility_radius
        # Get traffic lights
        traffic_signs = self.world.get_actors().filter('*traffic.traffic_light*')
        traffic_signs_loc = []
        step_ahead = 6.0
        for traffic_sign in traffic_signs:
            if str(traffic_sign.get_state()) != "Green":
                trigger = traffic_sign.trigger_volume
                traffic_sign.get_transform().transform(trigger.location)
                sign_w = self.map.get_waypoint(trigger.location, lane_type=carla.LaneType.Driving)
                theta = math.radians(sign_w.transform.rotation.yaw)
                b_x = trigger.location.x + step_ahead * math.cos(theta)
                b_y = trigger.location.y + step_ahead * math.sin(theta)
                trigger.location.x = b_x
                trigger.location.y = b_y
                traffic_signs_loc.append(trigger.location)

        # Get traffic stop signs
        traffic_signs = self.world.get_actors().filter('*traffic.stop*')
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
            traffic_signs_loc.append(trigger.location)

        traffic_signs_loc_temp = traffic_signs_loc
        traffic_signs_loc = []
        for loc_i in traffic_signs_loc_temp:
            if loc_i.distance(curr_location) < max_dist:
                traffic_signs_loc.append(loc_i)

        return [PathWaypoints(None, t_sign.x, t_sign.y) for t_sign in traffic_signs_loc]

    def find_all_driving_paths(self, vehicle_x, vehicle_y, max_path_length=100, points_dist=1):
        """
        Method to find the available driving paths to drive from the current vehicle point, it does't
        include paths from lane changes.
        :param vehicle_y: Vehicle y coordinate
        :param vehicle_x: Vehicle x coordinate
        :param max_path_length: the maximum length of each path
        :param points_dist: the distance of two consecutive waypoints
        :return final_paths: a list of lists of waypoints, each list is referred to a driving path
        :return separation_points: a list of tuples (cur_w, w_new) of the current and the next waypoint
                                    of each separation of the paths. The current waypoint is the junction point.
        """
        paths = [[]]
        final_paths = []
        closest_waypoint = self.map.get_waypoint(carla.Location(x=vehicle_x, y=vehicle_y),
                                                 lane_type=carla.LaneType.Driving)
        paths[0].append(closest_waypoint)
        path_length = [0.0]
        separation_points = []

        # Until at least one paths is into length range
        while len(paths) > 0:
            for id_path in range(len(paths)):  # for these paths
                path = paths[id_path]
                cur_w = path[-1]
                next_waypoints = cur_w.next(points_dist)
                w_in_radius = []
                for next_w in next_waypoints:  # find next waypoints
                    if path_length[id_path] + cur_w.transform.location.distance(
                            next_w.transform.location) < max_path_length:  # if new waypoints is into length range add them to a list
                        w_in_radius.append(next_w)
                if len(w_in_radius) == 1:  # if only one waypoint exists add to the end of the path
                    path.append(w_in_radius[0])
                    path_length[id_path] += cur_w.transform.location.distance(w_in_radius[0].transform.location)
                elif len(w_in_radius) > 1:  # if more than one waypoint then copy the path and add them to the list
                    for w_new in w_in_radius[1:]:
                        separation_points.append((PathWaypoints(cur_w), PathWaypoints(w_new)))
                        new_path_length = path_length[id_path] + cur_w.transform.location.distance(
                            w_new.transform.location)
                        path_length.append(new_path_length)
                        new_path = path[:]
                        new_path.append(w_new)
                        paths.append(new_path)
                    path.append(w_in_radius[0])
                    path_length[id_path] += cur_w.transform.location.distance(w_in_radius[0].transform.location)
                    separation_points.append((PathWaypoints(cur_w), PathWaypoints(w_in_radius[0])))
                else:  # if new waypoints don't exist them save path in final list and remove the path from the paths list
                    final_paths.append(path)
                    paths.remove(path)
                    path_length.remove(path_length[id_path])
                    break
        paths_list = []
        for f_p in final_paths:
            paths_list.append([])
            for path_waypoint in f_p:
                paths_list[-1].append(PathWaypoints(path_waypoint))
        return paths_list

    def find_lane_boarders(self, path_waypoints):
        """
        Method to find the left and right boarders of a lane or path.
        :param path_waypoints: A list of waypoints of the center of a lane, represent the base path that the vehicle
         has to follow
        :return left_border: And instance of the class Border that represent the left border of the lane
        :return right_border: And instance of the class Border that represent the left border of the lane
        """
        draw_points_flag = False or GLOBAL_DRAW
        left_border = Border()
        right_border = Border()

        for w_i in range(len(path_waypoints) - 1):
            # X_Y potion of boarders
            cur_w = path_waypoints[w_i]
            next_w = path_waypoints[w_i + 1]
            lane_width = cur_w.lane_width
            x1 = cur_w.transform.location.x
            y1 = cur_w.transform.location.y
            dx = next_w.transform.location.x - x1
            dy = next_w.transform.location.y - y1
            theta = math.atan2(dy, dx)
            theta_right = theta + math.pi / 2
            theta_left = theta - math.pi / 2
            right_border.x.append(x1 + (lane_width / 2) * math.cos(theta_right))
            right_border.y.append(y1 + (lane_width / 2) * math.sin(theta_right))
            left_border.x.append(x1 + (lane_width / 2) * math.cos(theta_left))
            left_border.y.append(y1 + (lane_width / 2) * math.sin(theta_left))

            # Lane type
            if cur_w.right_lane_marking.type is not None:
                right_border.lane_change.append(True)
            else:
                right_border.lane_change.append(False)

            if cur_w.left_lane_marking.type is not None:
                left_border.lane_change.append(True)
            else:
                left_border.lane_change.append(False)
        if draw_points_flag:
            self.draw_paths([[carla.Location(x=x, y=y) for x, y in zip(right_border.x, right_border.y)]],
                            life_time=5.2, color=[250, 0, 0], same_color=True, symbol="R")

            self.draw_paths([[carla.Location(x=x, y=y) for x, y in zip(left_border.x, left_border.y)]],
                            life_time=5.2, color=[250, 0, 0], same_color=True, symbol="L")

        return left_border, right_border


    def the_next_n_border_points(self, path_waypoints, left_border, right_border, n=2):
        """
        Method return the n next border point from the vehicle location and after .
        :param path_waypoints: A list of waypoints of the center of a lane, represent the base path that the vehicle
         has to follow
        :param right_border: And instance of the class Border that represent the left border of the lane
        :param left_border: And instance of the class Border that represent the left border of the lane
        :param n: The number of border points
        :return left_border: The left border has a length of N closer to the vehicle points
        :return right_border: The right border has a length of N closer to the vehicle points
        """

        if self.path_index + n < left_border.length():
            l_border = left_border.getFromTo(self.path_index, self.path_index + n)
        else:
            l_border = left_border.getFromTo(self.path_index, left_border.length() - 1)

        if self.path_index + n < right_border.length():
            r_border = right_border.getFromTo(self.path_index, self.path_index + n)
        else:
            r_border = right_border.getFromTo(self.path_index, right_border.length() - 1)

        return l_border, r_border

    def objects_around_the_vehicle(self, object_type="*vehicle*"):
        """
        Method to find all the objects of a type like vehicles, pedestrians,etc around
        the car in a certain distance defined by visibility radius
        :param object_type: String with the name of obstacle type like 'vehicle' 'pedestrian'
        :return target_objects: Carla actors objects
        """

        veh_loc = self.vehicle.get_location()
        actor_list = self.world.get_actors()
        objects_list = actor_list.filter(object_type)
        target_objects = []
        for an_object in objects_list:
            if veh_loc.distance(an_object.get_location()) < self.visibility_radius and \
                    self.vehicle.id != an_object.id and abs(an_object.get_location().z - veh_loc.z) < 5:
                target_objects.append(an_object)

        return target_objects

    def get_dynamic_objects(self, obj_type_list="vehicle", get_diff=False):
        """
        Method to return a instance of the class Objects with the objects around the car
        :param obj_type_list: A list of names with the objects that we want to find, the list can be
        like [vehicles,pedestrians] in order to retrieve all vehicles and pedestrians around the car. If obj_type
        includes the "all" statement then all the dynamic objects are saved and retrieved.
        :param get_diff: True to return the change of values because of noise
        :return objects: A dictionary with the dynamic objects
        """
        if not isinstance(obj_type_list, list):
            obj_type_list = [obj_type_list]
        if "all" in obj_type_list:
            obj_type_list = ["vehicle", "pedestrian", "traffic.traffic_light", "traffic.stop", "traffic.speed_limit"]

        objects = []
        for obj_type in obj_type_list:
            obj_list = self.objects_around_the_vehicle("*" + obj_type + "*")
            if len(obj_list) != 0:
                for object_i in obj_list:
                    objects.append(Objects(obj_type, object_i))
        # Add noise
        if get_diff:
            initial_obj_x = [obj.x for obj in objects]
            initial_obj_y = [obj.y for obj in objects]
            initial_obj_speed = [obj.speed for obj in objects]
        if self.p_d_add_noise:
            noise_x = np.random.normal(self.p_d_noise_mean, self.p_d_noise_std, len(objects))
            noise_y = np.random.normal(self.p_d_noise_mean, self.p_d_noise_std, len(objects))
            noise_yaw = np.random.normal(self.p_d_noise_mean, self.p_d_noise_std, len(objects))
            for i, object_i in enumerate(objects):
                object_i.x += noise_x[i]
                object_i.y += noise_y[i]
                object_i.yaw += noise_yaw[i]
        if self.s_add_noise:
            noise_vel_x = np.random.normal(self.s_noise_mean, self.s_noise_std, len(objects))
            noise_vel_y = np.random.normal(self.s_noise_mean, self.s_noise_std, len(objects))
            for i, object_i in enumerate(objects):
                object_i.vel_x += noise_vel_x[i]
                object_i.vel_y += noise_vel_y[i]
                object_i.speed = math.hypot(object_i.vel_y, object_i.vel_x)
        if get_diff:
            values_change = []
            for i in range(len(objects)):
                dist = round(math.hypot(objects[i].y - initial_obj_y[i], objects[i].x - initial_obj_x[i]), 2)
                speed = round(3.6*(objects[i].speed - initial_obj_speed[i]), 2)
                values_change.append([dist, speed])
            return objects, values_change
        return objects

    def get_route_curve(self, dist=60.0, regions_num=4):
        """
        The method calculate the curvature of a number of regions in the route path some distance ahead of the vehicle.
        Returns the average curvature of each region. As default we define 4 region in 100 meters ahead so return 4 values
        of curvature for each region.
        :param dist: The distance ahead of the vehicle to calculate the curvature
        :param regions_num: The number of the regions
        :return: The average curvature of each region
        """
        if self.route_path is not None:
            distance_ahead = 0
            index = self.path_index
            path_end_point = len(self.route_path) - 1
            curvature_list = []
            points_num = []
            dist_list = [0.0]
            for i in range(regions_num):
                curvature_list.append(0.0)
                points_num.append(0.0)
                dist_list.append((i + 1) * dist / regions_num)
            while distance_ahead < dist:
                curr_w = self.route_path[index]
                index += 1
                if index >= path_end_point or index + 1 >= path_end_point:
                    break
                next_w = self.route_path[index]
                relative_distance = math.hypot(next_w.y-curr_w.y, next_w.x-curr_w.x) + 0.0001
                distance_ahead += relative_distance
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

                curvature = angle_diff / relative_distance
                for ii in range(len(dist_list) - 1):
                    if dist_list[ii] <= distance_ahead <= dist_list[ii + 1]:
                        curvature_list[ii] += curvature
                        points_num[ii] += 1.0

            return [curvature_list[i] / points_num[i] if points_num[i] != 0 else 0.0 for i in range(regions_num)]
        else:
            return [0.0]

    def get_lanes_information(self):
        _, _, left_lane_closer = self.lane_change_availability()
        # Information for the current lane
        [ff_v, f_v, r_v, rr_v], [l_off_ff, l_off_f, l_off_r, l_off_rr], lane_width, f_n, r_n, opposite_direction = \
            self.get_objects_inside_lane_and_lane_information("vehicle")
        self.current_Lane.front_front_vehicle = ff_v
        self.current_Lane.front_vehicle = f_v
        self.current_Lane.rear_vehicle = r_v
        self.current_Lane.rear_rear_vehicle = rr_v
        self.current_Lane.vehicle_num_front = f_n
        self.current_Lane.vehicle_num_rear = r_n
        self.current_Lane.lateral_offset_front_front = l_off_ff
        self.current_Lane.lateral_offset_front = l_off_f
        self.current_Lane.lateral_offset_rear = l_off_r
        self.current_Lane.lateral_offset_front_rear = l_off_rr
        self.current_Lane.lane_width = lane_width
        self.current_Lane.opposite_direction = opposite_direction or self.current_Lane.lane_type != "Driving"

        # Information for the left lane
        if self.left_Lane.availability:
            [ff_v, f_v, r_v, rr_v], [l_off_ff, l_off_f, l_off_r, l_off_rr], lane_width, f_n, r_n, opposite_direction = \
                self.get_objects_inside_lane_and_lane_information("vehicle", lane_waypoint=self.left_waypoint)
            self.left_Lane.front_front_vehicle = ff_v
            self.left_Lane.front_vehicle = f_v
            self.left_Lane.rear_vehicle = r_v
            self.left_Lane.rear_rear_vehicle = rr_v
            self.left_Lane.vehicle_num_front = f_n
            self.left_Lane.vehicle_num_rear = r_n
            self.left_Lane.lateral_offset_front_front = l_off_ff
            self.left_Lane.lateral_offset_front = l_off_f
            self.left_Lane.lateral_offset_rear = l_off_r
            self.left_Lane.lateral_offset_front_rear = l_off_rr
            self.left_Lane.lane_width = lane_width
            self.left_Lane.opposite_direction = opposite_direction or self.left_Lane.lane_type != "Driving"
        else:
            self.left_Lane.unavailable_lane()

        # Information for the right lane
        if self.right_Lane.availability:
            [ff_v, f_v, r_v, rr_v], [l_off_ff, l_off_f, l_off_r, l_off_rr], lane_width, f_n, r_n, opposite_direction = \
                self.get_objects_inside_lane_and_lane_information("vehicle", lane_waypoint=self.right_waypoint)
            self.right_Lane.front_front_vehicle = ff_v
            self.right_Lane.front_vehicle = f_v
            self.right_Lane.rear_vehicle = r_v
            self.right_Lane.rear_rear_vehicle = rr_v
            self.right_Lane.vehicle_num_front = f_n
            self.right_Lane.vehicle_num_rear = r_n
            self.right_Lane.lateral_offset_front_front = l_off_ff
            self.right_Lane.lateral_offset_front = l_off_f
            self.right_Lane.lateral_offset_rear = l_off_r
            self.right_Lane.lateral_offset_front_rear = l_off_rr
            self.right_Lane.lane_width = lane_width
            self.right_Lane.opposite_direction = opposite_direction or self.right_Lane.lane_type != "Driving"
        else:
            self.right_Lane.unavailable_lane()

        # If both right and left lanes have opposite direction and current lane type is not driving then choose the closer one as available
        if self.right_Lane.opposite_direction and self.left_Lane.opposite_direction and self.current_Lane.lane_type != "Driving":
            if left_lane_closer:
                self.left_Lane.availability = True
                [ff_v, f_v, r_v, rr_v], [l_off_ff, l_off_f, l_off_r,
                                         l_off_rr], lane_width, f_n, r_n, opposite_direction = \
                    self.get_objects_inside_lane_and_lane_information("vehicle", lane_waypoint=self.left_waypoint)
                self.left_Lane.front_front_vehicle = ff_v
                self.left_Lane.front_vehicle = f_v
                self.left_Lane.rear_vehicle = r_v
                self.left_Lane.rear_rear_vehicle = rr_v
                self.left_Lane.vehicle_num_front = f_n
                self.left_Lane.vehicle_num_rear = r_n
                self.left_Lane.lateral_offset_front_front = l_off_ff
                self.left_Lane.lateral_offset_front = l_off_f
                self.left_Lane.lateral_offset_rear = l_off_r
                self.left_Lane.lateral_offset_front_rear = l_off_rr
                self.left_Lane.lane_width = lane_width
                self.left_Lane.opposite_direction = False
            else:
                self.right_Lane.availability = True
                [ff_v, f_v, r_v, rr_v], [l_off_ff, l_off_f, l_off_r,
                                         l_off_rr], lane_width, f_n, r_n, opposite_direction = \
                    self.get_objects_inside_lane_and_lane_information("vehicle", lane_waypoint=self.right_waypoint)
                self.right_Lane.front_front_vehicle = ff_v
                self.right_Lane.front_vehicle = f_v
                self.right_Lane.rear_vehicle = r_v
                self.right_Lane.rear_rear_vehicle = rr_v
                self.right_Lane.vehicle_num_front = f_n
                self.right_Lane.vehicle_num_rear = r_n
                self.right_Lane.lateral_offset_front_front = l_off_ff
                self.right_Lane.lateral_offset_front = l_off_f
                self.right_Lane.lateral_offset_rear = l_off_r
                self.right_Lane.lateral_offset_front_rear = l_off_rr
                self.right_Lane.lane_width = lane_width
                self.right_Lane.opposite_direction = False

        if GLOBAL_DRAW_LANES_VEHICLES_ARROWS or GLOBAL_DRAW:
            color = [0, 0, 0]
            self.draw_arrows_between_vehicles(self.current_Lane, color, life_time=DRAW_TIME)
            # color = [0, 200, 200]
            self.draw_arrows_between_vehicles(self.left_Lane, color, life_time=DRAW_TIME)
            # color = [0, 200, 200]
            self.draw_arrows_between_vehicles(self.right_Lane, color, life_time=DRAW_TIME)

        return self.left_Lane, self.current_Lane, self.right_Lane

    def lane_change_availability(self):
        veh_loc = self.vehicle.get_location()
        cur_w = self.map.get_waypoint(veh_loc, lane_type=carla.LaneType.Any)
        min_vehicle_width = 2.4  # 2.4m minimum space for the vehicle to pass a lane
        half_lane_width = cur_w.lane_width / 2.0
        offset = half_lane_width / 2.0
        lane_distance = half_lane_width + offset
        x1 = cur_w.transform.location.x
        y1 = cur_w.transform.location.y
        theta = math.radians(cur_w.transform.rotation.yaw)
        theta_right = theta + math.pi / 2.0
        theta_left = theta - math.pi / 2.0
        right_lane_x = x1 + lane_distance * math.cos(theta_right)
        right_lane_y = y1 + lane_distance * math.sin(theta_right)
        left_lane_x = x1 + lane_distance * math.cos(theta_left)
        left_lane_y = y1 + lane_distance * math.sin(theta_left)
        self.current_Lane.availability = True
        self.current_Lane.lane_type = str(cur_w.lane_type)

        right_waypoint = self.map.get_waypoint(
            carla.Location(x=right_lane_x, y=right_lane_y, z=cur_w.transform.location.z),
            lane_type=carla.LaneType.Any)

        left_waypoint = self.map.get_waypoint(
            carla.Location(x=left_lane_x, y=left_lane_y, z=cur_w.transform.location.z),
            lane_type=carla.LaneType.Any)

        # print(left_waypoint.lane_width)
        # Check if the autonomous vehicle is in a lane with opposite direction then the waypoints have to been swapped
        opposite_direction_flag = math.cos(math.radians(cur_w.transform.rotation.yaw -
                                                        self.vehicle.get_transform().rotation.yaw)) < 0.0
        if opposite_direction_flag:
            temp = right_waypoint
            right_waypoint = left_waypoint
            left_waypoint = temp

        if right_waypoint.transform.location.distance(cur_w.transform.location) > half_lane_width and \
                right_waypoint.lane_width > min_vehicle_width:
            self.right_waypoint = right_waypoint
            self.right_Lane.availability = True
            self.right_Lane.lane_type = str(right_waypoint.lane_type)
        else:
            self.right_waypoint = None
            self.right_Lane.availability = False

        if left_waypoint.transform.location.distance(cur_w.transform.location) > half_lane_width and \
                left_waypoint.lane_width > min_vehicle_width:
            self.left_waypoint = left_waypoint
            self.left_Lane.availability = True
            self.left_Lane.lane_type = str(left_waypoint.lane_type)
        else:
            self.left_waypoint = None
            self.left_Lane.availability = False

        # Check which direction is closer to a driving lane
        closer_waypoint = self.map.get_waypoint(carla.Location(x=x1, y=y1, z=cur_w.transform.location.z),
                                                lane_type=carla.LaneType.Driving)
        rel_x = closer_waypoint.transform.location.x - x1
        rel_y = closer_waypoint.transform.location.y - y1
        theta = math.degrees(math.atan2(rel_y, rel_x)) % 360.0
        theta = theta - self.vehicle.get_transform().rotation.yaw
        theta = theta % 360.0
        if theta > 180:
            left_lane_closer = True
        else:
            left_lane_closer = False

        if GLOBAL_DRAW_LANES_AVAILABILITY or GLOBAL_DRAW and False:
            if self.right_Lane.availability:
                w = right_waypoint
                location_text = w.transform.location
                location_text.z += 2
                self.world.debug.draw_string(location_text,
                                             " 0 " + " lane_type: = " + str(w.lane_type), draw_shadow=False,
                                             color=carla.Color(r=255, g=220, b=0), life_time=DRAW_TIME,
                                             persistent_lines=True)
            if self.left_Lane.availability:
                w = left_waypoint
                location_text = w.transform.location
                location_text.z += 2
                self.world.debug.draw_string(location_text,
                                             " 0 " + " lane_type: = " + str(w.lane_type), draw_shadow=False,
                                             color=carla.Color(r=255, g=220, b=0), life_time=DRAW_TIME,
                                             persistent_lines=True)

            location_text = cur_w.transform.location
            location_text.z += 1
            self.world.debug.draw_string(location_text,
                                         "X" + " lane_type: = " + str(cur_w.lane_type), draw_shadow=False,
                                         color=carla.Color(r=255, g=220, b=0), life_time=DRAW_TIME,
                                         persistent_lines=True)

        return self.left_Lane.availability, self.right_Lane.availability, left_lane_closer

    def get_objects_inside_lane_and_lane_information(self, objects_type="vehicle", meter_ahead_behind=50.0, lane_waypoint=None):
        """
        Method to return a instance of the class Objects with the two obstacles in front of and two
        behind the autonomous vehicle which they are in the same lane with the autonomous vehicle
        or the 'lane_waypoint', if there aren't any then the function returns None.
        :param objects_type: The type of objects to take into consideration, as default the type is "vehicles"
        :param meter_ahead_behind: Searching distance in front of the vehicle
        :param lane_waypoint: A waypoint of a specific lane to check if there are the front and behind vehicle's
        considering the autonomous' car direction. Is used for searching the adjacent lanes right and left of vehicle
        :return front_front_object: The object(vehicle,pedestrian etc) in front of the front vehicle of the autonomous
        vehicle
        :return front_object: The object(vehicle,pedestrian etc) in front of the autonomous vehicle
        :return rear_object: The object(vehicle,pedestrian etc) behind of the autonomous vehicle
        """
        min_dist_front_front = float("inf")
        min_dist_front = float("inf")
        min_dist_rear = float("inf")
        min_dist_rear_rear = float("inf")
        front_front_object = None
        front_object = None
        rear_object = None
        rear_rear_object = None
        num_of_front_vehicles = 0
        num_of_rear_vehicles = 0
        step = 1.0
        draw_points_flag = GLOBAL_DRAW_LANES_REPRESENTATION or GLOBAL_DRAW  # Visual representations of the waypoints on simulator
        objects_list = self.get_dynamic_objects(obj_type_list=objects_type)

        if lane_waypoint is None:
            veh_loc = self.vehicle.get_location()
            veh_x = veh_loc.x
            veh_y = veh_loc.y
            veh_closest_waypoint = self.map.get_waypoint(veh_loc, lane_type=carla.LaneType.Any)
        else:
            veh_closest_waypoint = lane_waypoint
            veh_x = veh_closest_waypoint.transform.location.x
            veh_y = veh_closest_waypoint.transform.location.y

        vehicle_lane_type = veh_closest_waypoint.lane_type
        lane_width = veh_closest_waypoint.lane_width
        max_lateral_distance = lane_width / 2.0
        # Check if the autonomous vehicle is in a lane with opposite direction
        opposite_direction_flag = math.cos(math.radians(veh_closest_waypoint.transform.rotation.yaw -
                                                        self.vehicle.get_transform().rotation.yaw)) < 0.0
        if len(objects_list) == 0:
            return [Objects(None), Objects(None), Objects(None), Objects(None)], \
                   [min_dist_front_front, min_dist_front, min_dist_rear,
                    min_dist_rear_rear], lane_width, 0.0, 0.0, opposite_direction_flag

        for sign in [-1.0, 1.0]:
            prev_w = veh_closest_waypoint
            waypoint_list = [prev_w]
            s_dist = 0.0
            while s_dist < meter_ahead_behind:
                cur_w = prev_w
                theta = math.radians(cur_w.transform.rotation.yaw)
                b_x = prev_w.transform.location.x + sign * step * math.cos(theta)
                b_y = prev_w.transform.location.y + sign * step * math.sin(theta)
                prev_w_loc = carla.Location(x=b_x, y=b_y, z=cur_w.transform.location.z)
                prev_w = self.map.get_waypoint(prev_w_loc, lane_type=vehicle_lane_type)
                waypoints_dist = cur_w.transform.location.distance(prev_w.transform.location)
                if waypoints_dist < step / 2.0:
                    break
                waypoint_list.append(prev_w)
                s_dist = s_dist + waypoints_dist

            for object_i in objects_list:
                object_closest_waypoint = self.map.get_waypoint(carla.Location(x=object_i.x, y=object_i.y),
                                                                lane_type=carla.LaneType.Any)
                vehicle_is_into_lane_front_flag = False
                vehicle_is_into_lane_rear_flag = False
                for w in waypoint_list:
                    dx = w.transform.location.x - object_closest_waypoint.transform.location.x
                    dy = w.transform.location.y - object_closest_waypoint.transform.location.y
                    dist = math.sqrt(dx ** 2 + dy ** 2)

                    if dist < max_lateral_distance:
                        dx = object_i.x - veh_x
                        dy = object_i.y - veh_y
                        dist = math.sqrt(dx ** 2 + dy ** 2)
                        if opposite_direction_flag is True and sign == -1 or opposite_direction_flag is False and sign == 1:
                            vehicle_is_into_lane_front_flag = True
                            if dist < min_dist_front:
                                min_dist_front_front = min_dist_front
                                front_front_object = front_object
                                min_dist_front = dist
                                front_object = object_i
                            elif dist < min_dist_front_front and front_object is not object_i:
                                min_dist_front_front = dist
                                front_front_object = object_i
                        else:
                            vehicle_is_into_lane_rear_flag = True
                            if dist < min_dist_rear:
                                min_dist_rear_rear = min_dist_rear
                                rear_rear_object = rear_object
                                min_dist_rear = dist
                                rear_object = object_i
                            elif dist < min_dist_rear_rear and rear_object is not object_i:
                                min_dist_rear_rear = dist
                                rear_rear_object = object_i

                if vehicle_is_into_lane_front_flag:
                    num_of_front_vehicles += 1
                if vehicle_is_into_lane_rear_flag:
                    num_of_rear_vehicles += 1

            if draw_points_flag and (
                    opposite_direction_flag is True and sign == -1.0 or opposite_direction_flag is False and sign == 1.0):
                for i in range(len(waypoint_list) - 1):
                    self.world.debug.draw_arrow(waypoint_list[i].transform.location,
                                                waypoint_list[i + 1].transform.location, thickness=0.1, arrow_size=0.1,
                                                color=carla.Color(r=255, g=0, b=100),
                                                life_time=0.14, persistent_lines=True)
            elif draw_points_flag:
                for i in range(len(waypoint_list) - 1):
                    self.world.debug.draw_arrow(waypoint_list[i].transform.location,
                                                waypoint_list[i + 1].transform.location, thickness=0.1, arrow_size=0.1,
                                                color=carla.Color(r=255, g=100, b=10),
                                                life_time=0.14, persistent_lines=True)
        # Correction when a vehicle is exactly beside the vehicle
        if front_object is rear_object and front_object is not None:
            num_of_rear_vehicles -= 1
        [lateral_offset_front_front, lateral_offset_front, lateral_offset_rear, lateral_offset_rear_rear] = [0.0, 0.0,
                                                                                                             0.0, 0.0]
        if front_front_object is not None:
            cur_w = self.map.get_waypoint(carla.Location(x=front_front_object.x, y=front_front_object.y),
                                          lane_type=carla.LaneType.Any)
            theta = math.radians(cur_w.transform.rotation.yaw)
            b_x = prev_w.transform.location.x + step * math.cos(theta)
            b_y = prev_w.transform.location.y + step * math.sin(theta)
            next_w_loc = carla.Location(x=b_x, y=b_y, z=cur_w.transform.location.z)
            next_w = self.map.get_waypoint(next_w_loc, lane_type=vehicle_lane_type)
            x_mm, y_mm = front_front_object.x, front_front_object.y
            Ax, Ay = cur_w.transform.location.x, cur_w.transform.location.y
            Bx, By = next_w.transform.location.x, next_w.transform.location.y
            curve_side = np.sign((Bx - Ax) * (y_mm - Ay) - (By - Ay) * (x_mm - Ax))
            curve_side = 1.0 if curve_side == 0.0 else curve_side
            lateral_offset_front_front = curve_side * cur_w.transform.location.distance(
                carla.Location(x=front_front_object.x, y=front_front_object.y))
            front_front_object.object_type = "front_front_object"

        if front_object is not None:
            cur_w = self.map.get_waypoint(carla.Location(x=front_object.x, y=front_object.y),
                                          lane_type=carla.LaneType.Any)
            theta = math.radians(cur_w.transform.rotation.yaw)
            b_x = prev_w.transform.location.x + step * math.cos(theta)
            b_y = prev_w.transform.location.y + step * math.sin(theta)
            next_w_loc = carla.Location(x=b_x, y=b_y, z=cur_w.transform.location.z)
            next_w = self.map.get_waypoint(next_w_loc, lane_type=vehicle_lane_type)
            x_mm, y_mm = front_object.x, front_object.y
            Ax, Ay = cur_w.transform.location.x, cur_w.transform.location.y
            Bx, By = next_w.transform.location.x, next_w.transform.location.y
            curve_side = np.sign((Bx - Ax) * (y_mm - Ay) - (By - Ay) * (x_mm - Ax))
            curve_side = 1.0 if curve_side == 0.0 else curve_side
            lateral_offset_front_front = curve_side * cur_w.transform.location.distance(
                carla.Location(x=front_object.x, y=front_object.y))
            front_object.object_type = "front_object"

        if rear_object is not None:
            cur_w = self.map.get_waypoint(carla.Location(x=rear_object.x, y=rear_object.y),
                                          lane_type=carla.LaneType.Any)
            theta = math.radians(cur_w.transform.rotation.yaw)
            b_x = prev_w.transform.location.x + step * math.cos(theta)
            b_y = prev_w.transform.location.y + step * math.sin(theta)
            next_w_loc = carla.Location(x=b_x, y=b_y, z=cur_w.transform.location.z)
            next_w = self.map.get_waypoint(next_w_loc, lane_type=vehicle_lane_type)
            x_mm, y_mm = rear_object.x, rear_object.y
            Ax, Ay = cur_w.transform.location.x, cur_w.transform.location.y
            Bx, By = next_w.transform.location.x, next_w.transform.location.y
            curve_side = np.sign((Bx - Ax) * (y_mm - Ay) - (By - Ay) * (x_mm - Ax))
            curve_side = 1.0 if curve_side == 0.0 else curve_side
            lateral_offset_rear = curve_side * cur_w.transform.location.distance(
                carla.Location(x=rear_object.x, y=rear_object.y))
            rear_object.object_type = "rear_object"

        if rear_rear_object is not None:
            cur_w = self.map.get_waypoint(carla.Location(x=rear_rear_object.x, y=rear_rear_object.y),
                                          lane_type=carla.LaneType.Any)
            theta = math.radians(cur_w.transform.rotation.yaw)
            b_x = prev_w.transform.location.x + step * math.cos(theta)
            b_y = prev_w.transform.location.y + step * math.sin(theta)
            next_w_loc = carla.Location(x=b_x, y=b_y, z=cur_w.transform.location.z)
            next_w = self.map.get_waypoint(next_w_loc, lane_type=vehicle_lane_type)
            x_mm, y_mm = rear_rear_object.x, rear_rear_object.y
            Ax, Ay = cur_w.transform.location.x, cur_w.transform.location.y
            Bx, By = next_w.transform.location.x, next_w.transform.location.y
            curve_side = np.sign((Bx - Ax) * (y_mm - Ay) - (By - Ay) * (x_mm - Ax))
            curve_side = 1.0 if curve_side == 0.0 else curve_side
            lateral_offset_rear_rear = curve_side * cur_w.transform.location.distance(
                carla.Location(x=rear_rear_object.x, y=rear_rear_object.y))
            rear_rear_object.object_type = "rear_rear_object"

        front_front_object = Objects(None) if front_front_object is None else front_front_object
        front_object = Objects(None) if front_object is None else front_object
        rear_object = Objects(None) if rear_object is None else rear_object
        rear_rear_object = Objects(None) if rear_rear_object is None else rear_rear_object

        lane_vehicles = [front_front_object, front_object, rear_object, rear_rear_object]
        vehicles_lateral_offset_from_lane = [lateral_offset_front_front, lateral_offset_front,
                                             lateral_offset_rear, lateral_offset_rear_rear]

        return lane_vehicles, vehicles_lateral_offset_from_lane, lane_width, \
               num_of_front_vehicles, num_of_rear_vehicles, opposite_direction_flag

    def global_to_vehicle_coord(self, obj_transform):
        """
        Conversion of the global coordinates of a target object into the vehicle's coordinate system
        :param obj_transform: An object of type carla.Transform of the target object
        :return local_transform: The transform regarding the vehicle's coordinate system
        """
        local_transform = carla.Transform()
        veh_loc = self.vehicle.get_transform().location
        obj_loc = obj_transform.location
        veh_rot = self.vehicle.get_transform().rotation
        obj_rot = obj_transform.rotation
        # Translation matrix
        T = np.array([[1.0, 0.0, 0.0, -veh_loc.x],
                      [0.0, 1.0, 0.0, -veh_loc.y],
                      [0.0, 0.0, 1.0, -veh_loc.z],
                      [0.0, 0.0, 0.0, 1.0]]
                     )
        # Rotation matrices
        theta = 0.0  # Rotation around x axis is consider negligible
        Rx = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, math.cos(theta), -math.sin(theta), 0.0],
                       [0.0, math.sin(theta), math.cos(theta), 0.0],
                       [0.0, 0.0, 0.0, 1.0]]
                      )

        theta = math.radians(veh_rot.pitch)
        Ry = np.array([[math.cos(theta), 0.0, math.sin(theta), 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [-math.sin(theta), 0.0, math.cos(theta), 0.0],
                       [0.0, 0.0, 0.0, 1.0]]
                      )

        theta = -math.radians(veh_rot.yaw)
        Rz = np.array([[math.cos(theta), -math.sin(theta), 0, 0],
                       [math.sin(theta), math.cos(theta), 0, 0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]]
                      )

        R = np.matmul((np.matmul(Rx, Ry)), Rz)

        '''
        a = -math.radians(veh_rot.yaw)
        b = math.radians(veh_rot.pitch)
        c = math.radians(veh_rot.roll)
        R = np.array([[math.cos(a) * math.cos(b), math.cos(a) * math.sin(b) * math.sin(c) - math.sin(a) * math.cos(c),
                       math.cos(a) * math.sin(b) * math.cos(c) + math.sin(a) * math.sin(c), 0],
                      [math.sin(a) * math.cos(b), math.sin(a) * math.sin(b) * math.sin(c) + math.cos(a) * math.cos(c),
                       math.sin(a) * math.sin(b) * math.cos(c) - math.cos(a) * math.sin(c), 0],
                      [-math.sin(b), math.cos(b) * math.sin(c), math.cos(b) * math.cos(c), 0],
                      [0, 0, 0, 1]])
        '''

        T_total = np.matmul(R, T)
        # Compute coordinates
        new_coordinates = np.matmul(T_total, [obj_loc.x, obj_loc.y, obj_loc.z, 1])

        local_transform.location.x = new_coordinates[0]
        local_transform.location.y = new_coordinates[1]
        local_transform.location.z = new_coordinates[2]
        local_transform.rotation.pitch = obj_rot.pitch - veh_rot.pitch
        local_transform.rotation.yaw = obj_rot.yaw - veh_rot.yaw
        local_transform.rotation.roll = obj_rot.roll - veh_rot.roll

        return local_transform

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

    def draw_arrows_between_vehicles(self, lane, color, life_time):
        lane_available = lane.availability
        carl_color1 = carla.Color(r=(25 + color[0]) % 255, g=(0 + color[1]) % 255, b=(100 + color[2]) % 255)
        carl_color2 = carla.Color(r=(254 + color[0]) % 255, g=(0 + color[1]) % 255, b=(100 + color[2]) % 255)
        carl_color3 = carla.Color(r=(254 + color[0]) % 255, g=(100 + color[1]) % 255, b=(10 + color[2]) % 255)
        carl_color4 = carla.Color(r=(254 + color[0]) % 255, g=(100 + color[1]) % 255, b=(100 + color[2]) % 255)

        if lane_available:
            ff_v, f_v = lane.front_front_vehicle, lane.front_vehicle
            b_v, bb_v = lane.rear_vehicle, lane.rear_rear_vehicle
            loc_veh = self.vehicle.get_location()
            loc_veh.z = loc_veh.z + 0.2

            if ff_v.object_type not in ["None"]:
                # print("Front")
                loc_front = carla.Location(x=ff_v.x, y=ff_v.y)
                loc_front.z = loc_veh.z + 0.2
                self.world.debug.draw_arrow(loc_veh, loc_front, thickness=0.1, arrow_size=0.1,
                                            color=carl_color1,
                                            life_time=life_time, persistent_lines=True)
            if f_v.object_type not in ["None"]:
                # print("Front")
                loc_front = carla.Location(x=f_v.x, y=f_v.y)
                loc_front.z = loc_veh.z + 0.2
                self.world.debug.draw_arrow(loc_veh, loc_front, thickness=0.1, arrow_size=0.1,
                                            color=carl_color2,
                                            life_time=life_time, persistent_lines=True)

            if b_v.object_type not in ["None"]:
                loc_beh = carla.Location(x=b_v.x, y=b_v.y)
                loc_beh.z = loc_veh.z + 0.2
                self.world.debug.draw_arrow(loc_veh, loc_beh, thickness=0.1, arrow_size=0.1,
                                            color=carl_color3,
                                            life_time=life_time, persistent_lines=True)

            if bb_v.object_type not in ["None"]:
                loc_beh = carla.Location(x=bb_v.x, y=bb_v.y)
                loc_beh.z = loc_veh.z + 0.2
                self.world.debug.draw_arrow(loc_veh, loc_beh, thickness=0.1, arrow_size=0.1,
                                            color=carl_color4,
                                            life_time=life_time, persistent_lines=True)

    def draw_text(self, location_xyz, text, color_rgb=None, life_time=0.1):
        if color_rgb is None:
            color_rgb = [250, 0, 0]
        if not isinstance(location_xyz, carla.Location):
            location_xyz = carla.Location(x=location_xyz[0], y=location_xyz[1], z=location_xyz[2])
        if not isinstance(color_rgb, carla.Color):
            color_rgb = carla.Color(r=color_rgb[0], g=color_rgb[1], b=color_rgb[2])
        self.world.debug.draw_string(location_xyz, text, draw_shadow=False,
                                     color=color_rgb, life_time=life_time,
                                     persistent_lines=True)


def main():
    # ==============================================================================
    # -- imports -------------------------------------------------------------------
    # ==============================================================================

  pass


if __name__ == '__main__':
    main()
