#!/usr/bin/env python

"""
This module implements all the available maneuvers of the vehicle, like a depository of available behaviours of the
vehicle like line keeping, overtaking while moving, overtaking when the front car is stopped, cross a junction etc..
Module functions::

"""

# ==============================================================================
# --General imports ------------------------------------------------------------
# ==============================================================================

from time import sleep
import math
import copy
from maneuvers import *
from rules_assessment import *

# ==============================================================================
# -- ROS imports ---------------------------------------------------------------
# ==============================================================================
import rospy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from perception.msg import Object
from perception.msg import TrafficSigns
from perception.msg import Lane
from perception.msg import RoadLanes
from perception.msg import Float64List
from perception.msg import StringList
from route_planner.msg import RouteLocation
from route_planner.srv import LaneOffsetRoute, LaneOffsetRouteRequest, LaneOffsetRouteResponse
from maneuver_generator.msg import ManeuverDataROS
from prediction.msg import VehiclesCollisionEventList
from prediction.msg import PedestrianCollisionEventList
from std_msgs.msg import Float64
from std_msgs.msg import Bool

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================


class ManeuverGenerator:
    def __init__(self):
        self.speed_limit = 25.0
        # MANEUVERS is in module maneuver.py. RULES__NUMBER is in module rules_assessment,py
        self.num_of_groups = len(MANEUVERS)
        self.maneuvers_list = []
        self.num_of_maneuvers = []
        for group in range(self.num_of_groups):
            self.maneuvers_list.append({MANEUVER: Maneuver(MANEUVER, RULES__NUMBER, group) for MANEUVER in MANEUVERS[group]})
            self.num_of_maneuvers.append(len(self.maneuvers_list[group]))
        self.rules_assessment_list = [0.0] * RULES__NUMBER
        self.time_duration = [0.0, 0.0]
        # Behavior data set
        self.maneuver_data = [ManeuverDataManGen()]*self.num_of_groups
        self.final_maneuver_data = ManeuverDataManGen()
        self.hold_current_maneuver = [False] * self.num_of_groups
        self.current_maneuver = [None] * self.num_of_groups
        # Overtake
        self.overtaking_object = None
        self.current_needed_offset = 3.0
        self.recovery_position = 0.0
        # Time to hold a maneuver
        self.maneuver_hold_time = 3.0
        # StopAndWait
        self.vehicle_stopped = False
        # --- ROS ---
        rospy.init_node('ManeuverGenerator_node', anonymous=True)
        self.ego_vehicle = Object()
        self.current_simulator_time = 0.0
        self.previous_time = [self.current_simulator_time] * 4
        self.route_curvature = []
        self.vehicles_collision_info = []
        self.pedestrians_collision_info = []
        self.left_lane = Lane()
        self.current_lane = Lane()
        self.right_lane = Lane()
        self.traffic_signs = TrafficSigns()
        self.lateral_offset_route = 0.0
        self.reach_end_of_route = False
        self.text_to_print = []
        self.subscriber_ego_vehicle = rospy.Subscriber("ego_vehicle_msg", Object, self.callback_ego_vehicle, queue_size=1)
        self.subscriber_curvature = rospy.Subscriber("curvature_msg", Float64List, self.callback_curvature, queue_size=1)
        self.subscriber_lanes_info = rospy.Subscriber("lanes_information_msg", RoadLanes, self.callback_lanes_info, queue_size=1)
        self.subscriber_traffic_signs = rospy.Subscriber("traffic_signs_msg", TrafficSigns, self.callback_traffic_signs, queue_size=1)
        self.subscriber_simulator_time_instance = rospy.Subscriber("simulator_time_instance_msg", Float64, self.callback_simulator_time_instance, queue_size=1)
        self.subscriber_vehicles_collision_info = rospy.Subscriber(
            "vehicles_collision_info_msg", VehiclesCollisionEventList, self.callback_vehicles_collision_info, queue_size=1)
        self.subscriber_pedestrians_collision_info = rospy.Subscriber(
            "pedestrians_collision_info_msg", PedestrianCollisionEventList, self.callback_pedestrians_collision_info, queue_size=1)
        self.subscriber_end_of_route = rospy.Subscriber("localization_in_route_path_msg", RouteLocation, self.callback_check_end_of_route, queue_size=1)
        self.pub_maneuver_data = rospy.Publisher('maneuver_data_msg', ManeuverDataROS, queue_size=1)
        self.pub_text_draw_maneuver = rospy.Publisher('text_draw_maneuver_msg', StringList, queue_size=1)

    def maneuvers_assessment(self):
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        if self.check_if_is_the_end_of_route_path():
            self.text_to_print[-1] = "True"
            self.publish_text_info_draw()
            return True
        self.text_to_print = []
        left_lane, current_lane, right_lane = copy.deepcopy(self.left_lane), copy.deepcopy(self.current_lane), copy.deepcopy(self.right_lane)
        curvature = copy.deepcopy(self.route_curvature)
        traffic_signs = copy.deepcopy(self.traffic_signs)
        vehicles_collision_events_info = copy.deepcopy(self.vehicles_collision_info)
        pedestrians_collision_event_info = copy.deepcopy(self.pedestrians_collision_info)
        if traffic_signs.speed_sign_exist and traffic_signs.speed_sign_distance < 10:
            self.speed_limit = traffic_signs.speed_limit-5
        else:
            traffic_signs.speed_limit = self.speed_limit

        # There is a vehicle front of the autonomous vehicle
        c0 = current_lane.front_vehicle.object_type not in ["None", ""]
        # There is available a lane on the left of the autonomous vehicle for driving on both directions
        c1 = left_lane.availability
        # There is available a lane on the left of the autonomous vehicle for driving with same direction
        c2 = left_lane.availability and not left_lane.opposite_direction
        # There is available a lane on the right of the autonomous vehicle for driving with same direction
        c3 = right_lane.availability and not right_lane.opposite_direction
        # Vehicle speed lower than speed limit
        c4 = math.hypot(ego_vehicle.vel_x, ego_vehicle.vel_y) < self.speed_limit
        # There is a vehicle behind the autonomous vehicle
        c5 = current_lane.rear_vehicle.object_type not in ["None", ""]
        # There are pedestrians
        c6 = len(pedestrians_collision_event_info) > 0
        # There are vehicles in scene
        c7 = len(vehicles_collision_events_info) > 0
        # Vehicle out of the road
        c8 = current_lane.lane_type != "Driving"
        # Vehicle not inside lane with opposite direction
        c9 = not current_lane.opposite_direction
        # Vehicle not inside lane with opposite direction
        c10 = current_lane.opposite_direction
        constraints = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
        active_rules_influence = [(1 if constraints[constraint_id] else 0) if constraint_id != -1 else 1
                                  for constraint_id in RULES_CONSTRAINTS_INFLUENCE]

        # High vehicle velocity relative to the front vehicle
        f0 = relative_velocity_front_vehicle_F0(ego_vehicle, current_lane.front_vehicle)
        self.rules_assessment_list[0] = f0

        # Coming vehicles from the left lane from the opposite direction(front)
        if c1:
            if left_lane.opposite_direction:
                f1 = opposite_direction_coming_vehicles_F1(ego_vehicle, left_lane.front_vehicle, left_lane.front_front_vehicle)
            else:
                f1 = 0.0
        else:
            f1 = 1.0
        self.rules_assessment_list[1] = f1

        # Coming vehicles from the left lane from the same direction(behind)
        if c2:
            if left_lane.opposite_direction is False:
                f2 = same_direction_coming_vehicles_F2_F11(ego_vehicle, left_lane.front_vehicle, left_lane.front_front_vehicle,
                                                       left_lane.rear_vehicle, left_lane.rear_rear_vehicle)
            else:
                f2 = 0.0
        else:
            f2 = 1.0
        self.rules_assessment_list[2] = f2

        # The lateral offset of the rear vehicle of the autonomous vehicle from its lane to the left side
        f3 = lateral_left_offset_rear_vehicle_F3(current_lane.rear_vehicle, current_lane.lateral_offset_rear, current_lane.lane_width)
        self.rules_assessment_list[3] = f3

        # The free space front of the front vehicle of the autonomous vehicle
        traffic_junction_info = [traffic_signs.traffic_junction_exist, traffic_signs.traffic_junction_distance]
        f4 = free_space_front_of_the_front_vehicle_F4(ego_vehicle, current_lane.front_vehicle,
                                                      current_lane.front_front_vehicle, traffic_junction_info)
        self.rules_assessment_list[4] = f4

        # The magnitude of road curvature
        f5 = road_curvature_F5(curvature, ego_vehicle.speed)
        self.rules_assessment_list[5] = f5

        # Time of collision with front vehicle
        f6 = collision_time_with_front_vehicle_F6(ego_vehicle, current_lane.front_vehicle)
        self.rules_assessment_list[6] = f6

        # Distance from front vehicle
        f7 = small_distance_from_front_vehicle_F7(ego_vehicle, current_lane.front_vehicle)
        self.rules_assessment_list[7] = f7

        # Low speed for much time
        # if is not in a traffic light  <---
        current_time = copy.deepcopy(self.current_simulator_time)
        f8, self.time_duration[0] = much_time_with_low_speed_F8(ego_vehicle, current_time, self.previous_time[0], self.time_duration[0])
        self.previous_time[0] = current_time
        self.rules_assessment_list[8] = f8

        # Left turn in sort distance
        f9 = left_turn_in_sort_distance_F9(curvature, ego_vehicle.speed)
        self.rules_assessment_list[9] = f9

        # Right turn in sort distance
        f10 = right_turn_in_sort_distance_F10(curvature, ego_vehicle.speed)
        self.rules_assessment_list[10] = f10

        # Coming vehicles from the right lane from the same direction(behind)
        if c3:
            if right_lane.opposite_direction is False:
                f11 = same_direction_coming_vehicles_F2_F11(ego_vehicle, right_lane.front_vehicle,
                                                       right_lane.front_front_vehicle,
                                                       right_lane.rear_vehicle, right_lane.rear_rear_vehicle)
            else:
                f11 = 0.0
        else:
            f11 = 1.0
        self.rules_assessment_list[11] = f11

        # The lateral offset of the rear vehicle of the autonomous vehicle from its lane to the right side
        f12 = lateral_right_offset_rear_vehicle_F12(current_lane.rear_vehicle, current_lane.lateral_offset_rear,
                                                 current_lane.lane_width)
        self.rules_assessment_list[12] = f12

        # Short distance from traffic light,stop sign, crosswalk.
        dist_from_sign = -1.0
        traffic_signs_type = "None"
        if traffic_signs.traffic_light_exist and traffic_signs.traffic_light_state in ["Red", "Yellow"]:
            dist_from_sign = traffic_signs.traffic_light_distance
            traffic_signs_type = traffic_signs.traffic_light_state

        if traffic_signs.stop_sign_exist:
            if (traffic_signs.traffic_light_exist and traffic_signs.stop_sign_distance < dist_from_sign) or \
                    not traffic_signs.traffic_light_exist:
                dist_from_sign = traffic_signs.stop_sign_distance
                traffic_signs_type = "Stop"

        if dist_from_sign > 0:
            f13 = short_distance_from_traffic_sign_F13(dist_from_sign, traffic_signs_type, ego_vehicle)
        else:
            f13 = 0.0
        self.rules_assessment_list[13] = f13

        # Vehicle speed lower than maximum limit
        f14 = vehicle_speed_lower_than_high_limit_F14(ego_vehicle, self.speed_limit)
        self.rules_assessment_list[14] = f14

        # Vehicle speed close to speed limit
        f15 = vehicle_speed_closer_to_desired_F15(ego_vehicle, self.speed_limit)
        self.rules_assessment_list[15] = f15

        # Vehicle speed higher than speed limit
        f16 = vehicle_speed_higher_than_high_limit_F16(ego_vehicle, self.speed_limit)
        self.rules_assessment_list[16] = f16

        # Less vehicles in the right lane
        f17 = less_vehicle_in_right_lane_F17(current_lane.vehicle_num_front, current_lane.vehicle_num_rear,
                                             right_lane.vehicle_num_front, right_lane.vehicle_num_rear)
        self.rules_assessment_list[17] = f17

        # Less vehicles in the left lane
        f18 = less_vehicle_in_left_lane_F18(current_lane.vehicle_num_front, current_lane.vehicle_num_rear,
                                            left_lane.vehicle_num_front, left_lane.vehicle_num_rear)
        self.rules_assessment_list[18] = f18

        # Less vehicles in the left lane
        collision_point = 0.0
        if c7:
            f19, collision_point = predict_short_collision_time_F19(ego_vehicle, vehicles_collision_events_info)
        else:
            f19 = 0.0
        self.rules_assessment_list[19] = f19

        # Low certainty for the prediction
        if c7:
            f20 = vehicle_prediction_probability_certaintyF20(vehicles_collision_events_info, collision_point)
        else:
            f20 = 0.0
        self.rules_assessment_list[20] = f20

        # Collision with pedestrian
        if c6:
            f21 = probability_pedestrian_collisionF21(ego_vehicle, pedestrians_collision_event_info)
        else:
            f21 = 0.0
        self.rules_assessment_list[21] = f21
        # Information to print in pygame window
        self.text_to_print.append(str(round(f20, 3)))
        self.text_to_print.append(str(round(f21, 3)))
        self.text_to_print.append(str(self.speed_limit+5))
        self.text_to_print.append("False")
        # Much time off the regular driving road
        current_time = copy.deepcopy(self.current_simulator_time)
        f22, self.time_duration[1] = much_time_off_driving_laneF22(current_lane.lane_type, current_time,
                                                                  self.previous_time[1], self.time_duration[1])
        self.previous_time[1] = current_time
        self.rules_assessment_list[22] = f22
        exceptions = []
        constraints = [1 if constraint is True else 0 for constraint in constraints]
        # Maneuvers which do not meat the constraints
        for group in range(self.num_of_groups):
            exceptions.append([])
            for MANEUVER in MANEUVERS[group]:
                maneuver_constraints = CONSTRAINTS[group][MANEUVERS[group].index(MANEUVER)]
                constraints_met = [i for i, x in enumerate(constraints) if x == C]
                constraints_mandatory = [i for i, x in enumerate(maneuver_constraints) if x == M]
                constraints_needed = [i for i, x in enumerate(maneuver_constraints) if x == C]
                constraints_direct_excepted = [i for i, x in enumerate(maneuver_constraints) if x == E]
                not_any_mandatory_const_met = not any(constraint in constraints_met for constraint in constraints_mandatory)
                not_all_needed_const_met = not all(constraint in constraints_met for constraint in constraints_needed)
                maneuver_direct_excepted = any(constraint in constraints_met for constraint in constraints_direct_excepted)
                there_are_constraints = len(constraints_mandatory) != 0
                if (there_are_constraints and (not_any_mandatory_const_met or not_all_needed_const_met)) or\
                        maneuver_direct_excepted:
                    exceptions[-1].append(MANEUVER)
        # Active maneuvers
        active_maneuvers = []
        for group in range(self.num_of_groups):
            active_maneuvers.append([MANEUVER for MANEUVER in MANEUVERS[group] if MANEUVER not in exceptions[group]])
            if len(active_maneuvers[group]) == 0:
                active_maneuvers[group] = SAFE_FAILURE_MANEUVER[group]
                print("------>  Safe Failure chosen maneuver :",  active_maneuvers[group])
        # Maneuvers assessment
        best_maneuvers = []
        best_assessment_value = []
        for group in range(self.num_of_groups):
            maneuvers_assessment_values = [[1]]
            while len(active_maneuvers[group]) > 1:
                maneuvers_assessment_values = [[self.maneuvers_list[group][MANEUVER].maneuver_assessment(
                        self.rules_assessment_list, group, active_maneuvers[group], active_rules_influence), MANEUVER]
                            for MANEUVER in active_maneuvers[group]]
                worst_maneuver = self.choose_worst_maneuver(group, maneuvers_assessment_values)
                active_maneuvers[group] = [maneuver for maneuver in active_maneuvers[group] if maneuver != worst_maneuver]
            #print("best_maneuver", active_maneuvers[group][0])
            best_maneuvers.append(active_maneuvers[group][0])
            best_assessment_value.append(max([v[0] for v in maneuvers_assessment_values]))
        # Apply best maneuvers
        for group in range(self.num_of_groups):
            if self.hold_current_maneuver[group] is not True:
                self.current_maneuver[group] = best_maneuvers[group]
        traffic_lanes_info = [left_lane, current_lane, right_lane]
        for group in range(self.num_of_groups):
            self.apply_maneuver(best_maneuvers[group], best_assessment_value[group], group, ego_vehicle,
                                traffic_lanes_info, traffic_signs, pedestrians_collision_event_info, current_time)
        self.extract_maneuver_data_regarding_the_maneuvers_combination()
        self.publish_maneuver_data()
        self.publish_text_info_draw()

    def apply_maneuver(self, maneuver_type, assessment_value, group, ego_vehicle, traffic_lanes_info, traffic_signs_info
                       , pedestrians_collision_event_info, current_time):
        angle_to_continue = 5.0
        assessment_value = assessment_value/30.0
        left_lane, current_lane, right_lane = traffic_lanes_info
        if pedestrians_collision_event_info:
            pedestrian = pedestrians_collision_event_info[0].object
            angle = pedestrians_collision_event_info[0].angle
            if abs(angle) > angle_to_continue:
                pedestrian = None
        else:
            pedestrian = None
        self.maneuver_data[group].direct_control = False
        maneuver_to_apply = maneuver_type
        if group == 0:
            # If we have to apply vehicle following then we reject the previous maneuver
            if maneuver_type == MANEUVERS[0][1]:
                self.hold_current_maneuver[group] = False
            if self.hold_current_maneuver[group]:
                maneuver_to_apply = self.current_maneuver[group]
            # < =======  Overtake  ======= >
            if maneuver_to_apply == MANEUVERS[0][0]:
                overtake_speed_diff = 20.0 / 3.6
                # Start maneuver
                if self.hold_current_maneuver[group] is False:
                    #print("Start maneuver")
                    self.hold_current_maneuver[group] = True
                    self.previous_time[3] = current_time
                    self.current_maneuver[group] = maneuver_to_apply
                    if current_lane.front_vehicle.object_type in ["None", ""]:
                        self.overtaking_object = pedestrian
                    elif pedestrian is not None:
                        dist_ped = math.hypot(ego_vehicle.y - pedestrian.y, ego_vehicle.x - pedestrian.x)
                        dist_veh = math.hypot(ego_vehicle.y - current_lane.front_vehicle.y, ego_vehicle.x - current_lane.front_vehicle.x)
                        if dist_ped < dist_veh:
                            self.overtaking_object = pedestrian
                        else:
                            self.overtaking_object = current_lane.front_vehicle
                    else:
                        self.overtaking_object = current_lane.front_vehicle
                    self.recovery_position = self.client_lane_offset_route(0)
                    self.current_needed_offset = self.client_lane_offset_route(-1)
                    self.maneuver_data[group], maneuver_ends = apply_overtake(ego_vehicle, self.overtaking_object,
                                                                              self.current_needed_offset,
                                                                              self.recovery_position, current_lane.lane_width,
                                                                              overtake_speed_diff, self.speed_limit)
                else:  # Hold maneuver
                    #print("Hold maneuver")
                    maneuver_ends = False
                    time_interval = current_time - self.previous_time[3]
                    if time_interval > self.maneuver_hold_time:
                        # Track objects position
                        if self.overtaking_object.object_type == "pedestrian":
                            if pedestrian is not None:
                                self.overtaking_object = pedestrian
                        else:
                            self.locate_overtaking_vehicle(traffic_lanes_info)
                        self.maneuver_data[group], maneuver_ends = apply_overtake(ego_vehicle, self.overtaking_object,
                                                                                  self.current_needed_offset,
                                                                                  self.recovery_position, current_lane.lane_width,
                                                                                  overtake_speed_diff, self.speed_limit)
                # Ends maneuver
                if maneuver_ends:
                    #print("End maneuver")
                    self.maneuver_data[group].left_road_width = -current_lane.lane_width
                    self.maneuver_data[group].right_road_width = current_lane.lane_width
                    self.hold_current_maneuver[group] = False

            # < =======  VehicleFollow  ======= >
            elif maneuver_to_apply == MANEUVERS[0][1]:
                self.hold_current_maneuver[group] = False
                self.current_maneuver[group] = maneuver_to_apply
                self.current_needed_offset = self.client_lane_offset_route(0)
                self.maneuver_data[group] = apply_vehicle_follow(ego_vehicle, current_lane.front_vehicle,
                                    self.current_needed_offset, current_lane.lane_width, self.speed_limit)

            # < =======  LeftLaneChange  ======= >
            elif maneuver_to_apply == MANEUVERS[0][2]:
                if self.hold_current_maneuver[group] is False:  # Start maneuver
                    time_interval = 0.0
                    self.current_needed_offset = self.client_lane_offset_route(-1)
                    self.previous_time[2] = current_time
                    self.hold_current_maneuver[group] = True
                    self.current_maneuver[group] = maneuver_to_apply
                else:
                    time_interval = current_time - self.previous_time[2]

                if time_interval > self.maneuver_hold_time or\
                        self.client_lane_offset_route(-1) != self.current_needed_offset:  # Ends maneuver
                    self.hold_current_maneuver[group] = False
                self.maneuver_data[group] = apply_left_lane_change(ego_vehicle, self.current_needed_offset,
                                                                   current_lane.lane_width)

            # < =======  RightLaneChange  ======= >
            elif maneuver_to_apply == MANEUVERS[0][3]:
                if self.hold_current_maneuver[group] is False:  # Start maneuver
                    time_interval = 0.0
                    self.current_needed_offset = self.client_lane_offset_route(1)
                    self.previous_time[2] = current_time
                    self.hold_current_maneuver[group] = True
                    self.current_maneuver[group] = maneuver_to_apply
                else:
                    time_interval = current_time - self.previous_time[2]

                if time_interval > self.maneuver_hold_time or\
                        self.client_lane_offset_route(1) != self.current_needed_offset:  # Ends maneuver
                    self.hold_current_maneuver[group] = False
                self.maneuver_data[group] = apply_right_lane_change(ego_vehicle, self.current_needed_offset,
                                                                    current_lane.lane_width)

            # < =======  FreeTravelStraight  ======= >
            elif maneuver_to_apply == MANEUVERS[0][4]:
                self.hold_current_maneuver[group] = False
                self.current_maneuver[group] = maneuver_to_apply
                self.current_needed_offset = self.client_lane_offset_route(0)
                self.maneuver_data[group] = apply_free_travel_straight(ego_vehicle, self.current_needed_offset,
                                                                       current_lane.lane_width)
        elif group == 1:
            # Check maneuver to apply
            if self.hold_current_maneuver[group]:
                maneuver_to_apply = self.current_maneuver[group]
            # < =======  Accelerate  ======= >
            if maneuver_to_apply == MANEUVERS[1][0]:
                self.hold_current_maneuver[group] = False
                self.current_maneuver[group] = maneuver_to_apply
                self.maneuver_data[group] = apply_acceleration(ego_vehicle.speed, traffic_signs_info.speed_limit, assessment_value)

            # < =======  Decelerate  ======= >
            elif maneuver_to_apply == MANEUVERS[1][1]:
                self.hold_current_maneuver[group] = False
                self.current_maneuver[group] = maneuver_to_apply
                self.maneuver_data[group] = apply_deceleration(ego_vehicle.speed, assessment_value)

            # < =======  SteadyState  ======= >
            elif maneuver_to_apply == MANEUVERS[1][2]:
                self.hold_current_maneuver[group] = False
                self.current_maneuver[group] = maneuver_to_apply
                self.maneuver_data[group] = apply_steady_state(ego_vehicle.speed)

                # < =======  StopAndWait  ======= >
            elif maneuver_to_apply == MANEUVERS[1][3]:
                # Start maneuver
                if self.hold_current_maneuver[group] is False and self.vehicle_stopped is False:
                    self.hold_current_maneuver[group] = True
                    self.current_maneuver[group] = maneuver_to_apply
                    self.maneuver_data[group], maneuver_ends = apply_stop_and_wait(ego_vehicle, traffic_lanes_info,
                                                                                   traffic_signs_info, pedestrian,
                                                                                   self.speed_limit)
                else:  # Hold maneuver
                    self.maneuver_data[group], clear_to_go = apply_stop_and_wait(ego_vehicle, traffic_lanes_info,
                                                                                traffic_signs_info, pedestrian,
                                                                                 self.speed_limit)
                    if self.vehicle_stopped is False:
                        self.vehicle_stopped = (ego_vehicle.speed < 1)
                    if self.vehicle_stopped and clear_to_go:
                        self.maneuver_data[group].target_speed = 4
                        self.hold_current_maneuver[group] = False
                    else:
                        self.vehicle_stopped = False
                    # print("-------------------------------stop", self.vehicle_stopped, clear_to_go)

                # Maneuver ends
                if not traffic_signs_info.stop_sign_exist and (pedestrian is None):
                    self.vehicle_stopped = False
                    self.hold_current_maneuver[group] = False

    def extract_maneuver_data_regarding_the_maneuvers_combination(self):
        group = 0
        """
        # Previous maneuver is "overtake" and current is "free travel straight"
        if self.final_maneuver_data.maneuver_type[0] == MANEUVERS[group][0] and self.current_maneuver[group] == MANEUVERS[group][4]:
            self.hold_current_maneuver[group] = True
            self.current_maneuver[group] = MANEUVERS[group][0]
        else:"""
        self.final_maneuver_data = self.maneuver_data[group].get_class_copy()
        if self.current_maneuver[group] not in [MANEUVERS[group][0], MANEUVERS[group][1]]:  # <-- Not Overtake or Vehicle follow
            self.final_maneuver_data.target_speed = self.maneuver_data[1].target_speed
        elif self.current_maneuver[group] in [MANEUVERS[group][1]]:  # <-- Vehicle follow
            if self.final_maneuver_data.target_speed > 5 and self.ego_vehicle.speed > 2:
                self.final_maneuver_data.target_speed = self.maneuver_data[1].target_speed

        group = 1
        self.final_maneuver_data.maneuver_type.append(self.maneuver_data[1].maneuver_type[0])
        if self.current_maneuver[group] in [MANEUVERS[group][3]]:
            self.final_maneuver_data.direct_control = self.maneuver_data[1].direct_control
            if self.maneuver_data[1].target_speed < self.final_maneuver_data.target_speed:
                self.final_maneuver_data.target_speed = self.maneuver_data[1].target_speed

        if self.final_maneuver_data.target_speed > self.speed_limit:
            self.final_maneuver_data.target_speed = self.speed_limit

    def choose_worst_maneuver(self, group, maneuvers_assessment_values):
        index = [value[0] for value in maneuvers_assessment_values].\
            index(min(value[0] for value in maneuvers_assessment_values))
        worst_maneuver = self.maneuvers_list[group][maneuvers_assessment_values[index][1]]
        return worst_maneuver.maneuver_type

    def locate_overtaking_vehicle(self, traffic_lanes_info):
        left_lane, current_lane, right_lane = traffic_lanes_info

        if current_lane.front_vehicle.object_type not in ["None", ""]:
            if current_lane.front_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = current_lane.front_vehicle
        elif current_lane.front_front_vehicle.object_type not in ["None", ""]:
            if current_lane.front_front_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = current_lane.front_front_vehicle
        elif current_lane.rear_vehicle.object_type not in ["None", ""]:
            if current_lane.rear_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = current_lane.rear_vehicle
        elif current_lane.rear_rear_vehicle.object_type not in ["None", ""]:
            if current_lane.rear_rear_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = current_lane.rear_rear_vehicle
        elif right_lane.front_vehicle.object_type not in ["None", ""]:
            if right_lane.front_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = right_lane.front_vehicle
        elif right_lane.front_front_vehicle.object_type not in ["None", ""]:
            if right_lane.front_front_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = right_lane.front_front_vehicle
        elif right_lane.rear_vehicle.object_type not in ["None", ""]:
            if right_lane.rear_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = right_lane.rear_vehicle
        elif right_lane.rear_rear_vehicle.object_type not in ["None", ""]:
            if right_lane.rear_rear_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = right_lane.rear_rear_vehicle
        elif left_lane.front_vehicle.object_type not in ["None", ""]:
            if left_lane.front_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = left_lane.front_vehicle
        elif left_lane.front_front_vehicle.object_type not in ["None", ""]:
            if left_lane.front_front_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = left_lane.front_front_vehicle
        elif left_lane.rear_vehicle.object_type not in ["None", ""]:
            if left_lane.rear_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = left_lane.rear_vehicle
        elif left_lane.rear_rear_vehicle.object_type not in ["None", ""]:
            if left_lane.rear_rear_vehicle.object_id == self.overtaking_object.object_id:
                self.overtaking_object = left_lane.rear_rear_vehicle
        else:
            self.overtaking_object = None

    def check_if_is_the_end_of_route_path(self):
        if self.reach_end_of_route:
            # When vehicle reach end of the route path then maneuvers -> ["TravelStraight", "StopAndWait"]
            self.current_maneuver = [MANEUVERS[0][4], MANEUVERS[1][3]]
            self.maneuver_data[0] = ManeuverDataManGen(lateral_offset=0.0, speed=0.0, maneuver_type=self.current_maneuver[0])
            self.maneuver_data[1] = ManeuverDataManGen(speed=0.0, direct_control=True, maneuver_type=self.current_maneuver[1])
            self.extract_maneuver_data_regarding_the_maneuvers_combination()
            self.publish_maneuver_data()
            self.reach_end_of_route = not self.ego_vehicle.speed == 0.0
            return True
        return False

    # -------- ROS functions ---------
    def publish_maneuver_data(self):
        pub = self.pub_maneuver_data
        maneuver_data_ros = ManeuverDataROS()
        maneuver_data_ros.maneuver_type = self.final_maneuver_data.maneuver_type
        maneuver_data_ros.dt = self.final_maneuver_data.dt
        maneuver_data_ros.from_time = self.final_maneuver_data.from_time
        maneuver_data_ros.to_time = self.final_maneuver_data.to_time
        maneuver_data_ros.time_sample_step = self.final_maneuver_data.time_sample_step
        maneuver_data_ros.direct_control = self.final_maneuver_data.direct_control
        maneuver_data_ros.target_lateral_offset = self.final_maneuver_data.target_lateral_offset
        maneuver_data_ros.left_road_width = self.final_maneuver_data.left_road_width
        maneuver_data_ros.right_road_width = self.final_maneuver_data.right_road_width
        maneuver_data_ros.num_of_paths_samples = self.final_maneuver_data.num_of_paths_samples
        maneuver_data_ros.target_speed = self.final_maneuver_data.target_speed  # m/s
        #rospy.loginfo(maneuver_data_ros)
        pub.publish(maneuver_data_ros)

    def publish_text_info_draw(self):
        if self.text_to_print:
            pub = self.pub_text_draw_maneuver
            text_data = StringList()
            text_data.string_list = self.text_to_print
            # rospy.loginfo(text_data)
            pub.publish(text_data)

    def callback_vehicles_collision_info(self, ros_data):
        self.vehicles_collision_info = ros_data.collision_event_list
        #rospy.loginfo(ros_data)

    def callback_pedestrians_collision_info(self, ros_data):
        self.pedestrians_collision_info = ros_data.collision_event_list
        #rospy.loginfo(ros_data)

    def callback_simulator_time_instance(self, ros_data):
        self.current_simulator_time = ros_data.data
        #rospy.loginfo(ros_data)

    def callback_ego_vehicle(self, ros_data):
        self.ego_vehicle = ros_data
        #rospy.loginfo(ros_data)

    def callback_curvature(self, ros_data):
        self.route_curvature = ros_data.float64_list
        #rospy.loginfo(ros_data)

    def callback_lanes_info(self, ros_data):
        self.left_lane = ros_data.left_lane
        self.current_lane = ros_data.current_lane
        self.right_lane = ros_data.right_lane
        #rospy.loginfo(ros_data)

    def callback_traffic_signs(self, ros_data):
        self.traffic_signs = ros_data
        #rospy.loginfo(ros_data)

    def callback_check_end_of_route(self, ros_data):
        self.reach_end_of_route = ros_data.end_of_path
        #rospy.loginfo(ros_data)

    def client_lane_offset_route(self, lane_id):
        rospy.wait_for_service('lane_offset_route_srv')
        try:
            lane_offset = rospy.ServiceProxy('lane_offset_route_srv', LaneOffsetRoute)
            resp1 = lane_offset(lane_id)
            #rospy.loginfo(resp1)
            return resp1.lane_offset_route
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

def main():
    #sleep(0.2)
    maneuver_generator = ManeuverGenerator()
    try:
        while not rospy.is_shutdown():
            maneuver_generator.maneuvers_assessment()
            #sleep(0.3)
    except rospy.ROSInterruptException:
        print(3)
        pass


if __name__ == '__main__':
    main()
