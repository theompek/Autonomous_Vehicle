#!/usr/bin/env python

"""
This module implements all the available maneuvers of the vehicle.
"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from time import sleep
import math
import os
import numpy as np

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================
# Each row is a group of maneuver that are interdependent but each group is independent of each other
MANEUVERS = [["Overtake", "VehicleFollow", "LeftLaneChange", "RightLaneChange", "TravelStraight"],
             ["Accelerate", "Decelerate", "SteadyState", "StopAndWait"]
             ]
SAFE_FAILURE_MANEUVER = [["TravelStraight"], ["Decelerate"]]  # <-- We can choose maneuver for safe failure mode
M, C, I, E = 2, 1, 0, -1
CONSTRAINTS = [
                [[M, C, I, I, C, I, M, I, I, I, I],
                 [M, I, I, I, I, I, I, I, I, C, E],
                 [M, I, C, I, I, I, M, I, M, I, M],
                 [M, I, I, C, I, I, M, I, M, I, M],
                 [E, I, I, I, I, I, I, I, I, I, E]],

                [[I, I, I, I, M, I, I, I, I, I, I],
                 [I, I, I, I, I, I, I, I, I, I, I],
                 [I, I, I, I, I, I, I, I, I, I, I],
                 [I, I, I, I, I, I, I, I, I, I, I]],
                ]
P, Z, N = 1, 0, -1
RULES_INFLUENCE_TABLE = [
                   [[P, N, N, N, P, N, P, P, P, N, N, P, P, N, P, P, P, N, P, P, P, P, P],
                    [N, P, P, P, N, P, N, N, N, N, N, P, P, P, P, P, P, N, N, N, P, N, N],
                    [P, N, N, N, Z, N, P, P, P, P, N, P, P, N, P, P, P, N, P, P, P, P, P],
                    [P, P, Z, P, Z, N, P, P, P, N, P, N, N, N, P, P, P, P, N, P, P, P, P],
                    [N, P, P, P, Z, P, N, N, Z, N, N, P, P, P, P, P, P, N, N, N, P, P, N]],

                   [[N, Z, Z, Z, Z, N, N, N, P, N, N, Z, Z, N, P, P, N, Z, Z, N, P, N, Z],
                    [P, Z, Z, Z, Z, P, P, P, Z, P, P, Z, Z, P, N, P, P, Z, Z, P, P, P, Z],
                    [N, Z, Z, Z, Z, P, N, N, Z, N, N, Z, Z, N, P, P, N, Z, Z, N, P, N, Z],
                    [P, Z, Z, Z, Z, P, P, P, N, P, P, Z, Z, P, N, Z, P, Z, Z, P, P, P, Z]]
                   ]
NAN, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9 , C10 = -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
RULES_CONSTRAINTS_INFLUENCE = [C0, C1, C1, C5, C0, NAN, C0, C0, NAN, NAN, NAN, C3, C5, NAN, NAN, NAN, NAN, C3, C2, C7, C7, C6, C8]
# RULES_WEIGHTS = [4, C1, C1, C5, C0, NAN, C0, C0, NAN, NAN, NAN, C3, C5, NAN, NAN, NAN, NAN, C3, C2]
MANEUVER_WEIGHTS = [
                   [[4, 6, 5, 1, 3, 2, 12, 7, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 25, 5, 1, 6, 1],
                    [3, 7, 6, 3, 20, 20, 6, 3, 2, 6, 6, 4, 3, 9, 5, 3, 3, 3, 3, 5, 8, 1, 1],
                    [2, 5, 4, 1, 0, 2, 7, 5, 2, 4, 1, 2, 1, 2, 2, 1, 1, 1, 4, 10, 1, 5, 10],
                    [2, 2, 0, 1, 0, 2, 7, 5, 2, 1, 4, 7, 4, 2, 2, 1, 1, 8, 1, 10, 1, 5, 10],
                    [2, 4, 4, 2, 0, 10, 4, 2, 0, 5, 5, 3, 2, 7, 4, 2, 2, 2, 2, 10, 10, 5, 1]],

                   [[5, 0, 0, 0, 0, 2, 4, 2, 2, 4, 4, 0, 0, 4, 10, 1, 10, 0, 0, 5, 1, 5, 0],
                    [8, 0, 0, 0, 0, 14, 12, 4, 0, 10, 10, 0, 0, 10, 10, 1, 10, 0, 0, 10, 10, 20, 0],
                    [1, 0, 0, 0, 0, 2, 1, 8, 0, 2, 2, 0, 0, 6, 4, 20, 4, 0, 0, 10, 5, 10, 0],
                    [2, 0, 0, 0, 0, 2, 6, 8, 1, 5, 5, 0, 0, 40, 2, 0, 2, 0, 0, 30, 10, 40, 0]]
                   ]
RULES__NUMBER = len(RULES_INFLUENCE_TABLE[0][0])


class ManeuverDataManGen:

    def __init__(self, left_width=-5.0, right_width=5.0, lateral_offset=None, speed=30.0, path_number=11, from_time=2.0,
                 to_time=3.0, time_step=1.0, dt=0.3, direct_control=False, maneuver_type=MANEUVERS[0][4]):
        self.maneuver_type = [maneuver_type]
        self.dt = dt
        # Time for traveling distance
        self.from_time = from_time
        self.to_time = to_time
        self.time_sample_step = time_step
        self.direct_control = direct_control
        # Position
        if abs(left_width) == abs(right_width):
            road_width = abs(right_width)
            if lateral_offset is not None:
                road_width = abs(lateral_offset) + 2.0 if (road_width - 2.0) < abs(lateral_offset) else road_width
            self.left_road_width = -road_width*1.0  # maximum road width to the left [m] <-- Maneuver
            self.right_road_width = road_width*1.0  # maximum road width to the right [m] <-- Maneuver
        else:
            self.left_road_width = left_width  # maximum road width to the left [m] <-- Maneuver
            self.right_road_width = right_width  # maximum road width to the right [m] <-- Maneuver
        self.num_of_paths_samples = path_number  # number of alternative paths <-- Compromise
        self.target_lateral_offset = lateral_offset if lateral_offset is not None else 0.0
        # Speed
        self.target_speed = speed / 3.6  # target speed [m/s] <-- Maneuver
        self.sampling_length = 0.15*speed / 3.6  # target speed sampling length [m/s] <-- Compromise

    def get_class_copy(self):
        return ManeuverDataManGen(right_width=self.right_road_width, left_width=self.left_road_width,
                                  lateral_offset=self.target_lateral_offset, speed=(self.target_speed*3.6),
                                  path_number=self.num_of_paths_samples, from_time=self.from_time, to_time=self.to_time,
                                  time_step=self.time_sample_step, dt=self.dt, direct_control=self.direct_control,
                                  maneuver_type=self.maneuver_type[0])


class Maneuver:
    def __init__(self, maneuver_type, rules_number, group):
        """
        Represent maneuver with information needed
        """
        self.maneuver_type = maneuver_type
        self.weights = self.cal_maneuver_weights_table(group)
        print("Maneuver Initialization: ", self.maneuver_type)
        print(self.maneuver_type, " weights: ", self.weights)

    def cal_maneuver_weights_table(self, group):
        weights = MANEUVER_WEIGHTS[group][MANEUVERS[group].index(self.maneuver_type)]
        # sum_w = sum(weights)
        # weights = [w/sum_w for w in weights]
        return weights

    def cal_maneuver_equal_weights(self, rules_number, group):
        weights = []
        index = MANEUVERS[group].index(self.maneuver_type)
        influence = RULES_INFLUENCE_TABLE[group][index]
        # Rules with zero effect
        excluded_rules = [i for i, x in enumerate(influence) if x == Z]
        num_of_excluded_rules = len(excluded_rules)
        rules_with_effect_number = rules_number-num_of_excluded_rules
        equal_weights = 1.0  # /rules_with_effect_number
        for i in range(rules_number):
            weights.append(equal_weights)
        for ex_rul in excluded_rules:
            weights[ex_rul] = 0.0
        return weights

    def maneuver_assessment(self, rules_values, group, active_maneuvers, active_rules_influence):
        assessment_value = 0.0
        index = MANEUVERS[group].index(self.maneuver_type)
        influence = RULES_INFLUENCE_TABLE[group][index][:]
        # Consider the rules which are the same for all the active maneuvers
        for active_maneuver in active_maneuvers:
            if active_maneuver != self.maneuver_type:
                id_active = MANEUVERS[group].index(active_maneuver)
                active_maneuver_influence = RULES_INFLUENCE_TABLE[group][id_active]
                for i in range(len(active_maneuver_influence)):
                    if active_maneuver_influence[i] == Z:
                        influence[i] = Z
        # Rules influence constraints
        for i in range(len(active_rules_influence)):
            if active_rules_influence[i] == Z:
                influence[i] = Z
        # Assessment of the maneuver
        weights = [self.weights[i] if influence[i] != Z else 0 for i in range(len(self.weights))]
        # sum_weights = sum(weights)
        # if sum_weights != 0:
            # weights = [w/sum_weights for w in weights]
        values_list = []
        for i in range(len(rules_values)):
            if influence[i] == P:
                assessment_value += weights[i] * rules_values[i]
                values_list.append(weights[i] * rules_values[i])
            elif influence[i] == N:
                assessment_value += weights[i] * (1-rules_values[i])
                values_list.append(weights[i] * (1-rules_values[i]))
            else:
                values_list.append(0)
                continue
        return assessment_value


def apply_overtake(ego_vehicle, overtaking_object, overtake_offset, initial_offset_from_route, lane_width, overtake_speed_diff=2.0, speed_limit=8.4):
    begin_to_overtake = False
    maneuver_ends = False
    min_lateral_dist = 3.0
    margin_on_the_edges = 2.0
    threshold_look_ahead_time = 5.0
    min_distance_to_overtake = 25.0
    min_distance_to_cancel = 6
    safe_recovery_dist = 20.0
    safe_speed_diff = 20.0  # m/s

    if overtaking_object is None:
        ego_vehicle_lateral_offset = initial_offset_from_route
        target_speed = ego_vehicle.speed
        maneuver_ends = True
    else:
        # Calculate relative velocity
        rel_vel_front_x = overtaking_object.vel_x - ego_vehicle.vel_x
        rel_vel_front_y = overtaking_object.vel_y - ego_vehicle.vel_y
        abs_value = math.sqrt(rel_vel_front_x ** 2.0 + rel_vel_front_y ** 2.0)  # m/s
        theta1 = math.degrees(math.atan2(rel_vel_front_y, rel_vel_front_x))
        theta1 = theta1 - ego_vehicle.yaw
        theta1 = math.radians(theta1 % 360.0)
        relative_velocity = -abs_value * math.cos(theta1)
        # Check if ego vehicle have overtake the front vehicle and then the maneuver ends
        relative_distance = math.hypot(overtaking_object.x - ego_vehicle.x, overtaking_object.y - ego_vehicle.y)
        relative_yaw = math.degrees(
            math.atan2(overtaking_object.y - ego_vehicle.y, overtaking_object.x - ego_vehicle.x))
        relative_yaw = relative_yaw % 360.0  # normalized in interval [0, 360)
        theta = relative_yaw - ego_vehicle.yaw
        theta = theta % 360.0  # normalized in interval [0, 360)
        if (130.0 < theta < 230.0 and ((relative_distance > safe_recovery_dist and  # <-- Front vehicle went behind ego vehicle
                relative_velocity > 0.0) or relative_velocity > safe_speed_diff)) or \
                ((theta < 20.0 or theta > 340.0) and relative_distance < min_distance_to_cancel):
            target_speed = ego_vehicle.speed
            ego_vehicle_lateral_offset = initial_offset_from_route
            maneuver_ends = True
        else:  # <-- Keep holding the  maneuver
            # Speed control
            time_to_collide = relative_distance/relative_velocity if relative_velocity > 0.0 else 2.0*threshold_look_ahead_time
            if 0.0 < time_to_collide < threshold_look_ahead_time or relative_distance < min_distance_to_overtake:  # <-- Do overtake
                begin_to_overtake = True
                target_speed = overtaking_object.speed + overtake_speed_diff
            else:  # <-- Wait to approach front vehicle
                target_speed = overtaking_object.speed + overtake_speed_diff / 2.0
            if 130.0 < theta:  # Return the vehicle to its initial position
                begin_to_overtake = False
                target_speed = overtaking_object.speed + overtake_speed_diff
            target_speed = ego_vehicle.speed if ego_vehicle.speed > target_speed else target_speed
            if math.sin(math.radians(theta)) * relative_distance < min_lateral_dist and ego_vehicle.speed > overtake_speed_diff / 2.0:
                target_speed = ego_vehicle.speed
            # Position control
            ego_vehicle_lateral_offset = initial_offset_from_route
            if begin_to_overtake:
                ego_vehicle_lateral_offset = overtake_offset
    if target_speed > (speed_limit + 3.0):
        target_speed = speed_limit + 3.0

    # Maneuver data
    distance_ahead = 15.0
    min_speed = 3.0  # 3m/s
    low_speed_time = 5.0
    high_speed_time = 2.0
    to_max_time = distance_ahead / target_speed if target_speed > min_speed else low_speed_time
    to_max_time = high_speed_time if to_max_time < high_speed_time else to_max_time
    time_step = 0.2
    from_min_time = to_max_time - time_step
    target_speed = target_speed * 3.6  # Km/h
    road_width = ego_vehicle_lateral_offset + margin_on_the_edges
    road_width = lane_width/2.0 + margin_on_the_edges if road_width < lane_width else road_width
    maneuver_data = ManeuverDataManGen(right_width=road_width, left_width=-road_width,
                                       lateral_offset=ego_vehicle_lateral_offset, speed=target_speed, path_number=11,
                                       from_time=from_min_time, to_time=to_max_time, time_step=time_step, dt=0.2,
                                       maneuver_type=MANEUVERS[0][0])
    return maneuver_data, maneuver_ends


def apply_vehicle_follow(ego_vehicle, front_vehicle, lateral_offset, lane_width, speed_limit):
    direct_control = True
    reaction_time = 2.0
    min_safety_distance = 4.0
    margin_on_the_edges = 2.0
    desired_distance = min_safety_distance + reaction_time*ego_vehicle.speed
    kd = 0.12
    ku = 0.2
    delay_t = 3.0
    speed_limit = speed_limit/3.6  # km/h -> m/s

    if front_vehicle is not None:
        # Calculate relative velocity
        rel_vel_front_x = front_vehicle.vel_x - ego_vehicle.vel_x
        rel_vel_front_y = front_vehicle.vel_y - ego_vehicle.vel_y
        abs_value = math.sqrt(rel_vel_front_x ** 2 + rel_vel_front_y ** 2)  # m/s
        theta1 = math.degrees(math.atan2(rel_vel_front_y, rel_vel_front_x))
        theta1 = theta1 - ego_vehicle.yaw
        theta1 = math.radians(theta1 % 360.0)
        relative_velocity = abs_value * math.cos(theta1)
        relative_distance = math.hypot(front_vehicle.x - ego_vehicle.x, front_vehicle.y - ego_vehicle.y)
        delta_d = relative_distance - desired_distance
        # delta_u = front_vehicle.speed - ego_vehicle.speed
        delta_u = relative_velocity
        acc_cmd = kd * delta_d + ku * delta_u
        target_speed = ego_vehicle.speed + acc_cmd * delay_t
        if target_speed < 2.0:
            target_speed = 0.0
        elif target_speed > speed_limit:
            target_speed = speed_limit
    else:
        direct_control = False
        target_speed = ego_vehicle.speed
    # Maneuver data
    distance_ahead = 15.0
    min_speed = 3.0  # 3m/s
    low_speed_time = 6.0
    high_speed_time = 2.0
    to_max_time = distance_ahead / target_speed if target_speed > min_speed else low_speed_time
    to_max_time = to_max_time if to_max_time > high_speed_time else high_speed_time
    time_step = 0.2
    from_min_time = to_max_time - time_step
    target_speed = target_speed * 3.6  # Km/h
    road_width = lateral_offset + margin_on_the_edges
    road_width = lane_width/2.0 + margin_on_the_edges if road_width < lane_width else road_width
    maneuver_data = ManeuverDataManGen(right_width=road_width, left_width=-road_width,
                                       lateral_offset=lateral_offset, speed=target_speed, path_number=11,
                                       from_time=from_min_time, to_time=to_max_time, time_step=time_step, dt=0.2,
                                       direct_control=direct_control, maneuver_type=MANEUVERS[0][1])
    return maneuver_data


def apply_left_lane_change(ego_vehicle, lateral_offset, lane_width):
    target_speed = ego_vehicle.speed
    margin_on_the_edges = 2.0
    # Maneuver data
    distance_ahead = 15.0
    min_speed = 3.0  # 3m/s
    low_speed_time = 3.0
    high_speed_time = 2.0
    to_max_time = distance_ahead / target_speed if target_speed > min_speed else low_speed_time
    to_max_time = to_max_time if to_max_time > high_speed_time else high_speed_time
    time_step = 0.2
    from_min_time = to_max_time - time_step
    target_speed = target_speed * 3.6  # Km/h
    road_width = lateral_offset + margin_on_the_edges
    road_width = lane_width/2.0 + margin_on_the_edges if road_width < lane_width else road_width
    maneuver_data = ManeuverDataManGen(right_width=road_width, left_width=-road_width,
                                       lateral_offset=lateral_offset, speed=target_speed, path_number=11,
                                       from_time=from_min_time, to_time=to_max_time, time_step=time_step, dt=0.2,
                                       maneuver_type=MANEUVERS[0][2])
    return maneuver_data


def apply_right_lane_change(ego_vehicle, lateral_offset, lane_width):
    target_speed = ego_vehicle.speed
    margin_on_the_edges = 2.0
    # Maneuver data
    distance_ahead = 15.0
    min_speed = 3.0  # 3m/s
    low_speed_time = 3.0
    high_speed_time = 2.0
    to_max_time = distance_ahead / target_speed if target_speed > min_speed else low_speed_time
    to_max_time = to_max_time if to_max_time > high_speed_time else high_speed_time
    time_step = 0.2
    from_min_time = to_max_time - time_step
    target_speed = target_speed * 3.6  # Km/h
    road_width = lateral_offset + margin_on_the_edges
    road_width = lane_width/2.0 + margin_on_the_edges if road_width < lane_width else road_width
    maneuver_data = ManeuverDataManGen(right_width=road_width, left_width=-road_width,
                                       lateral_offset=lateral_offset, speed=target_speed, path_number=11,
                                       from_time=from_min_time, to_time=to_max_time, time_step=time_step, dt=0.2,
                                       maneuver_type=MANEUVERS[0][3])
    return maneuver_data


def apply_free_travel_straight(ego_vehicle, lateral_offset, lane_width):
    target_speed = ego_vehicle.speed
    margin_on_the_edges = 2.0
    # Maneuver data
    distance_ahead = 15.0
    min_speed = 3.0  # 3m/s
    low_speed_time = 4.0
    high_speed_time = 2.0
    to_max_time = distance_ahead / target_speed if target_speed > min_speed else low_speed_time
    to_max_time = to_max_time if to_max_time > high_speed_time else high_speed_time
    time_step = 0.2
    from_min_time = to_max_time - time_step
    target_speed = target_speed * 3.6  # Km/h
    road_width = lateral_offset + margin_on_the_edges
    road_width = lane_width/2.0 + margin_on_the_edges if road_width < lane_width else road_width
    maneuver_data = ManeuverDataManGen(right_width=road_width, left_width=-road_width,
                                       lateral_offset=lateral_offset, speed=target_speed, path_number=11,
                                       from_time=from_min_time, to_time=to_max_time, time_step=time_step, dt=0.2,
                                       maneuver_type=MANEUVERS[0][4])
    return maneuver_data


def apply_acceleration(ego_vehicle_speed, speed_limit, acceleration_factor):
    speed_limit = speed_limit/3.6  # km/h -> m/s
    constant_acceleration = 2.0
    delay_factor = 2.0
    normalization_factor = 3.0
    d_u = (speed_limit - ego_vehicle_speed)/normalization_factor
    d_u = d_u if d_u > 0.0 else constant_acceleration
    target_speed = ego_vehicle_speed + d_u*acceleration_factor*delay_factor
    target_speed = target_speed*3.6
    return ManeuverDataManGen(speed=target_speed, maneuver_type=MANEUVERS[1][0])


def apply_deceleration(ego_vehicle_speed, deceleration_factor):
    delay_factor = 2.0
    normalization_factor = 7.0
    d_u = ego_vehicle_speed/normalization_factor
    target_speed = ego_vehicle_speed - d_u*deceleration_factor*delay_factor
    target_speed = target_speed if target_speed > 0.0 else 0.0
    target_speed = target_speed*3.6
    target_speed = 3.6 if ego_vehicle_speed < 2 else target_speed
    return ManeuverDataManGen(speed=target_speed, maneuver_type=MANEUVERS[1][1])


def apply_steady_state(ego_vehicle_speed):
    target_speed = ego_vehicle_speed*3.6
    return ManeuverDataManGen(speed=target_speed,  maneuver_type=MANEUVERS[1][2])


def apply_stop_and_wait(ego_vehicle, traffic_lanes_info, traffic_signs_info, pedestrian, speed_limit):
    direct_control = True
    clear_to_go = False
    wait_at_a_stop_sign = False
    wait_at_a_traffic_light = False
    stop_position_dist = 0.0
    dist_behind_stop_position = 2.0
    delay_factor = 2.0
    stop_area_radius = 4.0
    kd = 0.2
    ku = 0.53
    if traffic_signs_info.stop_sign_exist and traffic_signs_info.traffic_light_exist:
        stop_position_dist, wait_at_a_stop_sign = [traffic_signs_info.stop_sign_distance, True] if \
            traffic_signs_info.stop_sign_distance < traffic_signs_info.traffic_light_distance \
            else [traffic_signs_info.traffic_light_distance, False]
        wait_at_a_traffic_light = not wait_at_a_stop_sign
    elif traffic_signs_info.stop_sign_exist:
        stop_position_dist = traffic_signs_info.stop_sign_distance
        wait_at_a_stop_sign = True
        wait_at_a_traffic_light = False
    elif traffic_signs_info.traffic_light_exist:
        stop_position_dist = traffic_signs_info.traffic_light_distance
        wait_at_a_stop_sign = False
        wait_at_a_traffic_light = True
    wait_pedestrian = False
    if pedestrian is not None:
        dist_ped = math.hypot(ego_vehicle.y - pedestrian.y, ego_vehicle.x - pedestrian.x)
        if dist_ped < stop_position_dist:
            stop_position_dist = dist_ped
            wait_pedestrian = True

    delta_d = stop_position_dist - dist_behind_stop_position
    delta_u = -ego_vehicle.speed
    acc_cmd = kd * delta_d + ku * delta_u
    target_speed = ego_vehicle.speed + acc_cmd * delay_factor
    target_speed = target_speed if target_speed > 2.0 else 0.0
    target_speed = max(4, ego_vehicle.speed) if target_speed > ego_vehicle.speed else target_speed
    target_speed = target_speed * 3.6

    # In case of a stop sign we have to check for vehicles before continue
    safety_distance = 5.0
    if wait_at_a_stop_sign:
        left_lane, current_lane, right_lane = traffic_lanes_info
        left_d = 100000000.0
        current_d = 100000000.0
        right_d = 100000000.0
        if left_lane.front_vehicle is not None:
            left_d = math.hypot(left_lane.front_vehicle.y - ego_vehicle.y, left_lane.front_vehicle.x - ego_vehicle.x)
        if current_lane.front_vehicle is not None:
            current_d = math.hypot(current_lane.front_vehicle.y - ego_vehicle.y, current_lane.front_vehicle.x - ego_vehicle.x)
        if right_lane.front_vehicle is not None:
            right_d = math.hypot(right_lane.front_vehicle.y - ego_vehicle.y, right_lane.front_vehicle.x - ego_vehicle.x)
        if (left_lane.front_vehicle is None and current_lane.front_vehicle is None and right_lane.front_vehicle is None) \
                or (current_d > safety_distance and left_d > safety_distance and right_d > safety_distance):
            clear_to_go = True
    elif wait_pedestrian:
        clear_to_go = False
    elif wait_at_a_traffic_light:
        clear_to_go = True
    else:
        clear_to_go = True

    return ManeuverDataManGen(speed=target_speed, direct_control=direct_control, maneuver_type=MANEUVERS[1][3]), clear_to_go





