#!/usr/bin/env python

"""
This module implements all the cost functions for the assessment of the best maneuver
Module functions::

"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import math
import numpy as np

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================


def relative_velocity_front_vehicle_F0(ego_vehicle, front_vehicle):
    """
    Method to evaluate the F1 rule regarding the relative velocity between the autonomous vehicle
    and the front vehicle, we we suppose that they can have a maximum relative velocity. That rule describes the change
    in velocity of the autonomous car to reach the velocity of the front car

    """
    if front_vehicle.object_type in ["None", ""]:
        return 0.0
    max_vel_diff = 30.0  # maximum accepted difference in velocity from front vehicle in Km/h
    rel_vel_x = front_vehicle.vel_x - ego_vehicle.vel_x
    rel_vel_y = front_vehicle.vel_y - ego_vehicle.vel_y
    abs_value = math.sqrt(rel_vel_x ** 2 + rel_vel_y ** 2) * 3.6  # convert to Km/h
    theta = math.degrees(math.atan2(rel_vel_y, rel_vel_x))
    theta = theta - ego_vehicle.yaw
    theta = math.radians(theta % 360.0)
    relative_longitude_velocity = -abs_value * math.cos(theta)
    # relative_lateral_velocity = abs_value * math.sin(theta)
    k = 0.0 if abs(relative_longitude_velocity) < 0.1 else relative_longitude_velocity / max_vel_diff
    k = 1.0 if k > 1.0 else (0.0 if k < 0.0 else k)
    return k


def opposite_direction_coming_vehicles_F1(ego_vehicle, front_vehicle, front_front_vehicle):
    """
    Method to evaluate the F2 rule regarding the occupancy of the left lane from the coming vehicles from
    the opposite direction.

    """
    # Maximum vehicle velocity
    max_vehicle_velocity = 60.0  # 60 km/h
    # Maximum distance traveled by vehicle with the maximum speed
    distance_with_max_velocity = 20.0
    # Minimum distance from front vehicle in the left lane to suppose that the left lane is at the minimum safety level
    min_distance = 10.0
    # Maximum distance from front vehicle in the left lane to suppose that the left lane is at the maximum safety level
    max_distance = 50.0

    if front_vehicle.object_type not in ["None", ""]:
        rel_vel_front_x = front_vehicle.vel_x - ego_vehicle.vel_x
        rel_vel_front_y = front_vehicle.vel_y - ego_vehicle.vel_y
        abs_value = math.sqrt(rel_vel_front_x ** 2 + rel_vel_front_y ** 2) * 3.6  # convert to km/h
        theta = math.degrees(math.atan2(rel_vel_front_y, rel_vel_front_x))
        theta = theta - ego_vehicle.yaw
        theta = math.radians(theta % 360.0)
        relative_longitude_velocity_front = -abs_value * math.cos(theta)
        min_distance_front = min_distance + distance_with_max_velocity * (
                    relative_longitude_velocity_front / max_vehicle_velocity)
        max_distance_front = max_distance + distance_with_max_velocity * (
                    relative_longitude_velocity_front / max_vehicle_velocity)

        longitude_dist_front, lateral_dist_front = global_to_vehicle_coord(ego_vehicle.x, ego_vehicle.y,
                                                                           ego_vehicle.yaw, front_vehicle.x,
                                                                           front_vehicle.y)
        if longitude_dist_front < min_distance_front:
            k1 = 1.0
        elif longitude_dist_front > max_distance_front:
            k1 = 0.0
        else:
            k1 = 1.0 - (longitude_dist_front - min_distance_front) / (max_distance_front - min_distance_front)
    else:
        k1 = 0.0

    if front_front_vehicle.object_type not in ["None", ""]:
        rel_vel_front_front_x = front_front_vehicle.vel_x - ego_vehicle.vel_x
        rel_vel_front_front_y = front_front_vehicle.vel_y - ego_vehicle.vel_y
        abs_value = math.sqrt(rel_vel_front_front_x ** 2 + rel_vel_front_front_y ** 2) * 3.6  # convert to Km/h
        theta = math.degrees(math.atan2(rel_vel_front_front_y, rel_vel_front_front_x))
        theta = theta - ego_vehicle.yaw
        theta = math.radians(theta % 360.0)
        relative_longitude_velocity_front_front = -abs_value * math.cos(theta)
        min_distance += distance_with_max_velocity * (relative_longitude_velocity_front_front / max_vehicle_velocity)
        max_distance += distance_with_max_velocity * (relative_longitude_velocity_front_front / max_vehicle_velocity)
        longitude_dist_front_front, lateral_dist_front_front = global_to_vehicle_coord(ego_vehicle.x, ego_vehicle.y,
                                                                                       ego_vehicle.yaw, front_vehicle.x,
                                                                                       front_vehicle.y)
        if longitude_dist_front_front < min_distance:
            k2 = 1.0
        elif longitude_dist_front_front > max_distance:
            k2 = 0.0
        else:
            k2 = 1.0 - (longitude_dist_front_front - min_distance) / (max_distance - min_distance)
    else:
        k2 = 0.0

    k = k1 + k2

    return k if k < 1.0 else 1.0


def same_direction_coming_vehicles_F2_F11(ego_vehicle, front_vehicle, front_front_vehicle, rear_vehicle, rear_rear_vehicle):
    """
    Method to evaluate the F3 rule regarding the occupancy of the lanes from the coming vehicles from
    the same direction. Included vehicles which have overtake the autonomous vehicle and are front of it for some meters
    """

    # Maximum vehicle velocity
    max_vehicle_velocity = 60.0  # 60 Km/h
    # Maximum distance traveled by vehicle with the maximum speed
    distance_with_max_velocity = 20.0
    # Minimum distance from front vehicle to suppose the minimum safety level
    min_distance = 20.0
    # Maximum distance from front vehicle to suppose the maximum safety level
    max_distance = 30.0

    if front_vehicle.object_type not in ["None", ""]:
        rel_vel_front_x = front_vehicle.vel_x - ego_vehicle.vel_x
        rel_vel_front_y = front_vehicle.vel_y - ego_vehicle.vel_y
        abs_value = math.sqrt(rel_vel_front_x ** 2 + rel_vel_front_y ** 2) * 3.6  # convert to Km/h
        theta = math.degrees(math.atan2(rel_vel_front_y, rel_vel_front_x))
        theta = theta - ego_vehicle.yaw
        theta = math.radians(theta % 360.0)
        relative_longitude_velocity_front = -abs_value * math.cos(theta)
        min_distance_front = min_distance + distance_with_max_velocity * (
                    relative_longitude_velocity_front / max_vehicle_velocity)
        max_distance_front = max_distance + distance_with_max_velocity * (
                    relative_longitude_velocity_front / max_vehicle_velocity)
        longitude_dist_front, lateral_dist_front = global_to_vehicle_coord(ego_vehicle.x, ego_vehicle.y,
                                                                           ego_vehicle.yaw, front_vehicle.x,
                                                                           front_vehicle.y)
        longitude_dist_front = abs(longitude_dist_front)
        lateral_dist_front = abs(lateral_dist_front)
        if longitude_dist_front < min_distance_front:
            k1 = 1.0
        elif longitude_dist_front > max_distance_front:
            k1 = 0.0
        else:
            k1 = 1.0 - (longitude_dist_front - min_distance_front) / (max_distance_front - min_distance_front)
    else:
        k1 = 0.0

    if front_front_vehicle.object_type not in ["None", ""]:
        rel_vel_front_front_x = front_front_vehicle.vel_x - ego_vehicle.vel_x
        rel_vel_front_front_y = front_front_vehicle.vel_y - ego_vehicle.vel_y
        abs_value = math.sqrt(rel_vel_front_front_x ** 2 + rel_vel_front_front_y ** 2) * 3.6  # convert to Km/h
        theta = math.degrees(math.atan2(rel_vel_front_front_y, rel_vel_front_front_x))
        theta = theta - ego_vehicle.yaw
        theta = math.radians(theta % 360.0)
        relative_longitude_velocity_front_front = -abs_value * math.cos(theta)
        min_distance += distance_with_max_velocity * (relative_longitude_velocity_front_front / max_vehicle_velocity)
        max_distance += distance_with_max_velocity * (relative_longitude_velocity_front_front / max_vehicle_velocity)
        longitude_dist_front_front, lateral_dist_front_front = global_to_vehicle_coord(ego_vehicle.x, ego_vehicle.y,
                                                                                       ego_vehicle.yaw,
                                                                                       front_front_vehicle.x,
                                                                                       front_front_vehicle.y)
        longitude_dist_front_front = abs(longitude_dist_front_front)
        # lateral_dist_front_front = abs(lateral_dist_front_front)
        if longitude_dist_front_front < min_distance:
            k2 = 1.0
        elif longitude_dist_front_front > max_distance:
            k2 = 0.0
        else:
            k2 = 1.0 - (longitude_dist_front_front - min_distance) / (max_distance - min_distance)
    else:
        k2 = 0.0

    """
        Vehicles coming from behind the autonomous vehicle
    """
    # Maximum vehicle velocity
    max_vehicle_velocity = 60.0  # 60 Km/h
    # Maximum distance traveled by vehicle with the maximum speed
    distance_with_max_velocity = 20.0
    # Minimum distance from front vehicle in the left lane to suppose that the left lane is at the minimum safety level
    min_distance = 10.0
    # Maximum distance from front vehicle in the left lane to suppose that the left lane is at the maximum safety level
    max_distance = 50.0

    if rear_vehicle.object_type not in ["None", ""]:
        rel_vel_front_x = rear_vehicle.vel_x - ego_vehicle.vel_x
        rel_vel_front_y = rear_vehicle.vel_y - ego_vehicle.vel_y
        abs_value = math.sqrt(rel_vel_front_x ** 2 + rel_vel_front_y ** 2) * 3.6  # convert to Km/h
        theta = math.degrees(math.atan2(rel_vel_front_y, rel_vel_front_x))
        theta = theta - ego_vehicle.yaw
        theta = math.radians(theta % 360.0)
        relative_longitude_velocity_rear = abs_value * math.cos(theta)
        min_distance_rear = min_distance + distance_with_max_velocity * (
                relative_longitude_velocity_rear / max_vehicle_velocity)
        max_distance_rear = max_distance + distance_with_max_velocity * (
                relative_longitude_velocity_rear / max_vehicle_velocity)

        longitude_dist_rear, lateral_dist_rear = global_to_vehicle_coord(ego_vehicle.x, ego_vehicle.y,
                                                                         ego_vehicle.yaw, rear_vehicle.x,
                                                                         rear_vehicle.y)
        longitude_dist_rear = abs(longitude_dist_rear)
        lateral_dist_rear = abs(lateral_dist_rear)
        if longitude_dist_rear < min_distance_rear:
            k3 = 1.0
        elif longitude_dist_rear > max_distance_rear:
            k3 = 0.0
        else:
            k3 = 1.0 - (longitude_dist_rear - min_distance_rear) / (max_distance_rear - min_distance_rear)
    else:
        k3 = 0.0

    if rear_rear_vehicle.object_type not in ["None", ""]:
        rel_vel_front_front_x = rear_rear_vehicle.vel_x - ego_vehicle.vel_x
        rel_vel_front_front_y = rear_rear_vehicle.vel_y - ego_vehicle.vel_y
        abs_value = math.sqrt(rel_vel_front_front_x ** 2 + rel_vel_front_front_y ** 2) * 3.6  # convert to Km/h
        theta = math.degrees(math.atan2(rel_vel_front_front_y, rel_vel_front_front_x))
        theta = theta - ego_vehicle.yaw
        theta = math.radians(theta % 360.0)
        relative_longitude_velocity_rear_rear = abs_value * math.cos(theta)
        min_distance += distance_with_max_velocity * (relative_longitude_velocity_rear_rear / max_vehicle_velocity)
        max_distance += distance_with_max_velocity * (relative_longitude_velocity_rear_rear / max_vehicle_velocity)
        longitude_dist_rear_rear, lateral_dist_rear_rear = global_to_vehicle_coord(ego_vehicle.x, ego_vehicle.y,
                                                                                   ego_vehicle.yaw, rear_rear_vehicle.x,
                                                                                   rear_rear_vehicle.y)
        longitude_dist_rear_rear = abs(longitude_dist_rear_rear)
        lateral_dist_rear_rear = abs(lateral_dist_rear_rear)
        if longitude_dist_rear_rear < min_distance:
            k4 = 1.0
        elif longitude_dist_rear_rear > max_distance:
            k4 = 0.0
        else:
            k4 = 1.0 - (longitude_dist_rear_rear - min_distance) / (max_distance - min_distance)
    else:
        k4 = 0.0

    k = k1 + k2 + k3 + k4

    return k if k < 1.0 else 1.0


def lateral_left_offset_rear_vehicle_F3(rear_vehicle, lateral_offset_rear, lane_width):
    if lateral_offset_rear >= 0.0 or rear_vehicle.object_type  in ["None", ""]:
        return 0.0
    k = lane_width/2.0 - lane_width/8.0
    k = (abs(lateral_offset_rear)/k)**2.0
    return k if k < 1.0 else 1.0


def free_space_front_of_the_front_vehicle_F4(ego_vehicle, front_vehicle, front_front_vehicle, traffic_junction_info):
    # The minimum safe distance between the front and the front_front vehicle
    min_distance = 30.0
    # The distance between the front and the front_front vehicle to suppose the safety is ok
    max_distance = 45.0
    # Maximum vehicles velocity
    max_vehicle_velocity = 60.0  # 60 Km/h
    # Maximum distance traveled by vehicle with the maximum speed
    distance_with_max_velocity = 10.0

    if front_front_vehicle.object_type  in ["None", ""]:
        [traffic_junction_exist, traffic_junction_distance] = traffic_junction_info
        if (not traffic_junction_exist) or (front_vehicle.object_type  in ["None", ""]):
            return 1.0
        ego_front_dist = math.hypot(front_vehicle.x - ego_vehicle.x, front_vehicle.y - ego_vehicle.y)
        distance = traffic_junction_distance - ego_front_dist
        distance = 0.0 if distance < 0.0 else distance
        min_distance_front = min_distance
        max_distance_front = max_distance
    else:
        rel_vel_front_x = front_front_vehicle.vel_x - front_vehicle.vel_x
        rel_vel_front_y = front_front_vehicle.vel_y - front_vehicle.vel_y
        abs_value = math.sqrt(rel_vel_front_x ** 2 + rel_vel_front_y ** 2) * 3.6  # convert to Km/h
        theta = math.degrees(math.atan2(rel_vel_front_y, rel_vel_front_x))
        theta = theta - front_vehicle.yaw
        theta = math.radians(theta % 360.0)
        relative_longitude_velocity_front = -abs_value * math.cos(theta)
        min_distance_front = min_distance + distance_with_max_velocity * (relative_longitude_velocity_front / max_vehicle_velocity)
        max_distance_front = max_distance + distance_with_max_velocity * (relative_longitude_velocity_front / max_vehicle_velocity)
        distance = math.hypot(front_front_vehicle.x - front_vehicle.x, front_front_vehicle.y - front_vehicle.y)

    if distance < min_distance_front:
        k = 0.0
    elif distance > max_distance_front:
        k = 1.0
    else:
        k = (distance - min_distance_front) / (max_distance_front - min_distance_front)
    return k


def road_curvature_F5(curvature, ego_vehicle_speed):
    """
    The method estimate the magnitude of the road curvature using the curvature of a number of regions of the road
    ahead the vehicle. The closest to vehicle regions are more important so we use higher values of weights in an
    equation with weights for each region multiplied with the curvature value of the region
    :param curvature: The curvature of a number of regions of the road ahead the vehicle
    :param ego_vehicle_speed: Speed of ego vehicle
    :return:
    """
    max_curvature = 2.8
    max_safe_speed = 40.0
    ego_vehicle_speed = ego_vehicle_speed*3.6  # m/s -> km/h
    regions_num = len(curvature)
    curvature = [(0.0 if -0.01 < ci < 0.01 else ci) for ci in curvature]
    k1 = 0.0
    sum_weights = 0.00001
    for i in range(regions_num):
        k1 += (regions_num-i)*curvature[i]
        sum_weights += regions_num-i

    k1 = abs(k1/(sum_weights*max_curvature))
    k2 = ego_vehicle_speed/max_safe_speed
    k2 = 0 if ego_vehicle_speed < max_safe_speed else k2
    k = k1 * (1 + 5*k2)
    return k if k < 1.0 else 1.0


def collision_time_with_front_vehicle_F6(ego_vehicle, front_vehicle):
    if front_vehicle.object_type  in ["None", ""]:
        return 0.0
    min_distance_margin = 7.0  # 7 meters minimum between the vehicles for collision
    max_relative_velocity = 50.0  # 60 Km/h
    min_dist_with_max_vel = 30.0
    t_min = min_dist_with_max_vel/max_relative_velocity

    # Approximate the lateral distance with absolute distance
    dist = math.hypot(front_vehicle.x - ego_vehicle.x, front_vehicle.y - ego_vehicle.y)
    dist = (dist - min_distance_margin)
    if dist <= 0.0:
        return 1.0
    rel_vel_front_x = front_vehicle.vel_x - ego_vehicle.vel_x
    rel_vel_front_y = front_vehicle.vel_y - ego_vehicle.vel_y
    abs_value = math.sqrt(rel_vel_front_x ** 2 + rel_vel_front_y ** 2) * 3.6  # convert to Km/h
    theta = math.degrees(math.atan2(rel_vel_front_y, rel_vel_front_x))
    theta = theta - ego_vehicle.yaw
    theta = math.radians(theta % 360.0)
    relative_longitude_velocity_front = -abs_value * math.cos(theta)
    if relative_longitude_velocity_front <= 0.0:
        return 0.0

    t = dist/relative_longitude_velocity_front

    return t_min/t if t > t_min else 1.0


def small_distance_from_front_vehicle_F7(ego_vehicle, front_vehicle):
    if front_vehicle.object_type  in ["None", ""]:
        return 0.0
    min_distance_margin = 7.0  # 7 meters minimum between the vehicles for collision
    max_distance_margin = 15.0
    # Approximate the lateral distance with absolute distance
    dist = math.hypot(front_vehicle.x - ego_vehicle.x, front_vehicle.y - ego_vehicle.y)
    if dist <= min_distance_margin:
        return 1.0

    k = (max_distance_margin-dist)/(max_distance_margin-min_distance_margin)
    return 0.0 if k < 0.0 else k


def much_time_with_low_speed_F8(ego_vehicle, current_time, previous_time, low_speed_duration):
    min_speed = 2.0
    medium_speed = 4.0
    min_waiting_time = 30.0
    max_waiting_time = 50.0
    veh_speed = math.hypot(ego_vehicle.vel_x, ego_vehicle.vel_y)*3.6  # Km/h
    if veh_speed <= min_speed:
        low_speed_duration += current_time - previous_time
    elif min_speed < veh_speed < medium_speed:
        low_speed_duration -= current_time - previous_time
    else:
        low_speed_duration = 0.0

    if low_speed_duration < min_waiting_time:
        k = 0.0
    elif low_speed_duration > max_waiting_time:
        k = 1.0
    else:
        k = ((low_speed_duration-min_waiting_time)/(max_waiting_time-min_waiting_time))**2

    return k, low_speed_duration


def left_turn_in_sort_distance_F9(curvature, ego_vehicle_speed):
    max_curvature = 1.0
    max_safe_speed = 40.0
    ego_vehicle_speed = ego_vehicle_speed * 3.6  # m/s -> km/h
    regions_num = len(curvature)
    curvature = [(0.0 if -0.01 < ci < 0.01 else ci) for ci in curvature]
    k1 = 0.0
    sum_weights = 0.000001
    for i in range(regions_num):
        if (regions_num - i) * curvature[i] <= 0:
            k1 += (regions_num - i) * curvature[i]
            sum_weights += regions_num - i
    k1 = k1 / (sum_weights * max_curvature)
    k1 = k1 if k1 < 0.0 else 0.0
    k1 = abs(k1) if k1 > -1 else 1
    k2 = ego_vehicle_speed / max_safe_speed
    k2 = 0 if ego_vehicle_speed < max_safe_speed else k2
    k = k1 * (1 + 5*k2)
    return k if k < 1.0 else 1.0


def right_turn_in_sort_distance_F10(curvature, ego_vehicle_speed):
    max_curvature = 1.0
    max_safe_speed = 40.0
    ego_vehicle_speed = ego_vehicle_speed * 3.6  # m/s -> km/h
    regions_num = len(curvature)
    curvature = [(0.0 if -0.01 < ci < 0.01 else ci) for ci in curvature]
    k1 = 0.0
    sum_weights = 0.000001
    for i in range(regions_num):
        if (regions_num - i) * curvature[i] >= 0:
            k1 += (regions_num - i) * curvature[i]
            sum_weights += regions_num - i
    k1 = k1 / (sum_weights * max_curvature)
    k1 = k1 if k1 > 0.0 else 0.0
    k1 = k1 if k1 < 1.0 else 1.0
    k2 = ego_vehicle_speed / max_safe_speed
    k2 = 0 if ego_vehicle_speed < max_safe_speed else k2
    k = k1 * (1 + 5*k2)
    return k if k < 1.0 else 1.0


def lateral_right_offset_rear_vehicle_F12(rear_vehicle, lateral_offset_rear, lane_width):
    if lateral_offset_rear <= 0.0 or rear_vehicle.object_type  in ["None", ""]:
        return 0.0
    k = lane_width/2.0 - lane_width/8.0
    k = (lateral_offset_rear/k)**2
    return k if k < 1.0 else 1.0


def short_distance_from_traffic_sign_F13(dist_from_sign, traffic_signs_type, ego_vehicle):
    max_distance_light = 60.0
    min_distance_light = 8.0
    max_distance_stop = 60.0
    min_distance_stop = 4.0
    max_speed = 60.0/3.6  # 60km/h -> m/s
    max_distance_decelerate = 20.0  # m
    vehicle_speed = math.hypot(ego_vehicle.vel_x, ego_vehicle.vel_y)  # m/s
    t_min = max_distance_decelerate/max_speed
    min_distance_for_yellow = vehicle_speed*t_min

    if traffic_signs_type == "Yellow" and dist_from_sign < min_distance_for_yellow:
        k = 0.0
    elif traffic_signs_type in ["Red", "Yellow"]:
        if dist_from_sign < min_distance_light:
            k = 1.0
        elif dist_from_sign > max_distance_light:
            k = 0.0
        else:
            k = (max_distance_light - dist_from_sign) / (max_distance_light - min_distance_light)
    elif traffic_signs_type == "Stop":
        if dist_from_sign < min_distance_stop:
            k = 1.0
        elif dist_from_sign > max_distance_stop:
            k = 0.0
        else:
            k = (max_distance_stop - dist_from_sign) / (max_distance_stop - min_distance_stop)
    else:
        k = 0.0

    return k


def vehicle_speed_lower_than_high_limit_F14(ego_vehicle, speed_limit):
    speed_margin = -1.0
    vehicle_speed = math.hypot(ego_vehicle.vel_x, ego_vehicle.vel_y)*3.6  # km/h
    k = (vehicle_speed/(speed_limit+speed_margin))
    k = 1.0-k if k < 1.0 else 0.0
    return k


def vehicle_speed_closer_to_desired_F15(ego_vehicle, desired_speed):
    speed_margin = 5.0
    vehicle_speed = math.hypot(ego_vehicle.vel_x, ego_vehicle.vel_y)*3.6  # km/h
    if vehicle_speed < desired_speed:
        k = ((vehicle_speed - desired_speed + speed_margin)/speed_margin)
    else:
        k = ((desired_speed - vehicle_speed + speed_margin) / speed_margin)
    k = (0.0 if k < 0.0 else k) if k < 1.0 else 1.0
    return math.sqrt(k)


def vehicle_speed_higher_than_high_limit_F16(ego_vehicle, speed_limit):
    speed_margin = 10.0
    vehicle_speed = math.hypot(ego_vehicle.vel_x, ego_vehicle.vel_y)*3.6  # km/h
    k = ((vehicle_speed-speed_limit+speed_margin)/speed_margin)
    k = (k if k > 0.0 else 0.0) if k < 1.0 else 1.0
    return k**2


def less_vehicle_in_right_lane_F17(curr_num_front, curr_num_rear, right_num_front, right_num_rear):
    max_vehicle_front_diff = 3.0
    max_vehicle_rear_diff = 2.0
    w1 = 0.7
    w2 = 0.3
    if curr_num_front == right_num_front or curr_num_front == 0:
        w1 = 0.0
        w2 = 0.0
        k1 = 0.0
    elif curr_num_front < right_num_front:
        k1 = 0.0
    else:
        k1 = (curr_num_front - right_num_front)/max_vehicle_front_diff

    if curr_num_rear == right_num_rear:
        w1 = 1.0
        w2 = 0.0
        k2 = 0.0
    elif curr_num_rear < right_num_rear:
        k2 = 0.0
    else:
        k2 = (curr_num_rear-right_num_rear)/max_vehicle_rear_diff

    k = w1*k1 + w2*k2
    k = 1.0 if k > 1.0 else k
    return k


def less_vehicle_in_left_lane_F18(curr_num_front, curr_num_rear, left_num_front, left_num_rear):
    max_vehicle_front_diff = 3.0
    max_vehicle_rear_diff = 2.0
    w1 = 0.7
    w2 = 0.3
    if curr_num_front == left_num_front or curr_num_front == 0:
        w1 = 0.0
        w2 = 0.0
        k1 = 0.0
    elif curr_num_front < left_num_front:
        k1 = 0.0
    else:
        k1 = (curr_num_front - left_num_front) / max_vehicle_front_diff

    if curr_num_rear == left_num_rear:
        w1 = 1.0
        w2 = 0.0
        k2 = 0.0
    elif curr_num_rear < left_num_rear:
        k2 = 0.0
    else:
        k2 = (curr_num_rear - left_num_rear) / max_vehicle_rear_diff

    k = w1 * k1 + w2 * k2
    k = 1.0 if k > 1.0 else k
    return k


def predict_short_collision_time_F19(ego_vehicle, collision_events_info):
    r_min = 0.1
    t_react = 3.0
    d_react = 7.0
    vehicles = []
    collision_points = []
    collisions_time = []
    collision_probability = []
    for collision_events in collision_events_info:
        vehicles.append(collision_events.object)
        collision_points.append(collision_events.collision_point)
        collisions_time.append(collision_events.collision_time)

    for collision_time, vehicle, collision_point in zip(collisions_time, vehicles, collision_points):
        collision_time_probability = (1.0 - r_min) * math.exp(-0.5 * ((collision_time / t_react) ** 2)) + r_min
        dist_veh = math.hypot(ego_vehicle.x - vehicle.x, ego_vehicle.y - vehicle.y)
        collision_vehicles_dist_probability = (1.0 - r_min) * math.exp(-0.5 * ((dist_veh / d_react) ** 2)) + r_min
        dist_coll_point = math.hypot(ego_vehicle.x - collision_point.x, ego_vehicle.y - collision_point.y)
        collision_point_dist_probability = (1.0 - r_min) * math.exp(-0.5 * ((dist_coll_point / d_react) ** 2)) + r_min
        collision_probability.append(0.4*collision_time_probability +
                                     0.4*collision_vehicles_dist_probability + 0.2*collision_point_dist_probability)

    return max(collision_probability), collision_probability.index(max(collision_probability))


def vehicle_prediction_probability_certaintyF20(collision_events_info, collision_index):
    return collision_events_info[collision_index].prediction_probability


def probability_pedestrian_collisionF21(ego_vehicle, collision_events_info):
    r_min = 0.02
    t_react = 3.0
    d_lat_react = 4.0
    d_log_react = 5.0
    ange_react = 20.0
    collision_events_info = collision_events_info[0]
    pedestrian = collision_events_info.object
    angle = collision_events_info.angle
    coll_t = collision_events_info.collision_time
    coll_dist = collision_events_info.collision_distance
    collision_time_probability = (1.0 - r_min) * math.exp(-0.5 * ((coll_t / t_react) ** 2)) + r_min
    collision_lateral_dist_probability = (1.0 - r_min) * math.exp(-0.5 * ((coll_dist / d_lat_react) ** 2)) + r_min
    log_dist = math.hypot(ego_vehicle.y - pedestrian.y, ego_vehicle.x - pedestrian.x)
    collision_longitude_dist_probability = (1.0 - r_min) * math.exp(-0.5 * ((log_dist / d_log_react) ** 2)) + r_min
    collision_angle_probability = (1.0 - r_min) * math.exp(-0.5 * ((abs(angle) / ange_react) ** 2)) + r_min
    collision_probability = (0.25*collision_time_probability + 0.25*collision_longitude_dist_probability +
                             0.25*collision_lateral_dist_probability + 0.25*collision_angle_probability)
    return collision_probability


def much_time_off_driving_laneF22(lane_type, current_time, previous_time, time_duration):
    min_waiting_time = 2.0
    max_waiting_time = 4.0
    time_duration = 0.0 if time_duration < 0.0 else time_duration
    if lane_type != "Driving":
        time_duration += current_time - previous_time
    else:
        time_duration -= current_time - previous_time

    if time_duration < min_waiting_time:
        k = 0.0
    elif time_duration > max_waiting_time:
        k = 1.0
    else:
        k = ((time_duration - min_waiting_time) / (max_waiting_time - min_waiting_time)) ** 2

    return k, time_duration


def global_to_vehicle_coord(vehicle_x, vehicle_y, vehicle_yaw, obj_x, obj_y):
    """
    Conversion of the global coordinates of a target object into the vehicle's coordinate system
    """
    # Translation matrix
    T = np.array([[1.0, 0.0, -vehicle_x],
                  [0.0, 1.0, -vehicle_y],
                  [0.0, 0.0, 1.0]]
                 )
    # Rotation matrices
    theta = -math.radians(vehicle_yaw)
    Rz = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                   [math.sin(theta), math.cos(theta), 0.0],
                   [0.0, 0.0, 1.0]]
                  )

    T_total = np.matmul(Rz, T)
    # Compute coordinates
    new_coordinates = np.matmul(T_total, [obj_x, obj_y, 1])

    return new_coordinates[0], new_coordinates[1]
