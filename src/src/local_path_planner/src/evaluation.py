#!/usr/bin/env python

"""
The module using a base path as guider searches all possible paths around the vehicle toward to base path and finds the
best path for the car to follow
Module functions:
1) List of possible paths : A list of candidate paths for the vehicle to follow
2) Objects around the car: The position and orientation ot objects like other cars and pedestrian relative to our car


"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from time import sleep
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================


def vehicle_boundary_to_angle(vehicle_dimensions, angle):
    """
    Method that specifies imaginary boundary around the vehicle,it is defined as parts of ellipse for the front and
    the rear of the vehicle and part of lines laterally of the vehicle. The method returns the radial of the boundary
    from the center of the vehicle for specific angle calculated relative to the front of the vehicle.
    :param  vehicle_dimensions: The vehicle dimensions that the boundaries are calculated for
    :param  angle: The angle has values in degrees and the direction is anticlockwise,has a value of zero front of the
    vehicle,90 exactly to the left,180 behind the vehicle, 270 exactly to the right of the vehicle and so on for
    the intermediate values.
    """
    length = vehicle_dimensions.length
    width = vehicle_dimensions.width
    l_offset = 0.7
    w_offset = 1.0
    a = length + l_offset
    b = width + w_offset
    q1, q2, q3, q4 = 30.0, 150.0, 210.0, 330.0

    if q1 < angle < q2:
        r_q1 = a * b / np.hypot(a * math.sin(math.radians(q1)), b * math.cos(math.radians(q1)))
        d1 = r_q1 * math.sin(math.radians(q1))
        r = d1 / math.cos(math.radians(angle - 90.0))
    elif q3 < angle < q4:
        r_q3 = a * b / np.hypot(a * math.sin(math.radians(q3)), b * math.cos(math.radians(q3)))
        d2 = -r_q3 * math.sin(math.radians(q3))
        r = d2 / math.cos(math.radians(angle - 270.0))
    else:
        r = a * b / np.hypot(a * math.sin(math.radians(angle)), b * math.cos(math.radians(angle)))

    return r


def vehicle_boundaries(vehicle):
    """
    Method that specifies imaginary boundary around the vehicle,it is defined as parts of ellipse for the front and
    the rear of the vehicle and part of lines laterally of the vehicle. The method returns the radial distance of the
    boundaries from the center of the vehicle for every angle between of 0 and 360 degrees per one degree accuracy,
    so overall are returned 360 values of radial distances.
    :param  vehicle: The vehicle that the boundaries are calculated for
    """
    length = vehicle.bounding_box.extent.x
    width = vehicle.bounding_box.extent.y
    l_offset = 0.7
    w_offset = 0.7
    a = length + l_offset
    b = width + w_offset
    r = [a*b/np.hypot(a*math.sin(math.radians(fi)), b*math.cos(math.radians(fi))) for fi in range(360)]
    q1, q2, q3, q4 = 30.0, 150.0, 210.0, 330.0

    d1 = r[q1]*math.sin(math.radians(q1))
    d2 = -r[q3]*math.sin(math.radians(q3))
    for i in range(q1, q2):
        r[i] = d1/math.cos(math.radians(i-90))
    for i in range(q3, q4):
        r[i] = d2/math.cos(math.radians(i-270))

    return r


def close_to_base_path(candidate_paths):
    """
    Method to calculate the lateral distance of the end of the path from the base path. It returns an value normalized
    in the interval [0,1].
    :param  candidate_paths: The candidate paths given for evaluation
    """
    d = [abs(path.d[-1]) for path in candidate_paths]
    d_min = min(d)
    d = [di - d_min for di in d]
    d_max = max(d)
    return [float(1.0-di/d_max) for di in d]


def close_to_lateral_offset_target(candidate_paths, target_offset):
    """
    Method to evaluate how close the paths are to the target lateral offset point given by maneuver generator.
    It returns an value normalized in the interval [0,1].
    :param candidate_paths: The candidate paths given for evaluation
    :param  target_offset: The target offset from the route path
    """
    d = [abs(path.d[-1]-target_offset) for path in candidate_paths]
    d_min = min(d)
    d = [di - d_min for di in d]
    d_max = max(d)
    k = [float(1.0-di/d_max) for di in d]
    return k


def path_close_to_previous_one(candidate_paths, previous_optimal_path):
    """
    Method to choose a path close to previous optimal path avoiding  abrupt changes in vehicle direction.
    :param candidate_paths: The candidate paths given for evaluation
    :param  previous_optimal_path: The previous optimal path
    """
    prev_path_lateral_offset = previous_optimal_path.d[-1]
    d = [abs(path.d[-1] - prev_path_lateral_offset) for path in candidate_paths]
    d_min = min(d)
    d = [di - d_min for di in d]
    d_max = max(d)
    k = [float(1.0 - di / d_max) for di in d]
    return k


def far_away_from_objects(candidate_paths, objects, ego_vehicle):
    """
    Method that calculate the closest distance from the points of the path to the obstacles. We take into consideration
    the boundary form around the objects so as to the the objects do not collide with ego vehicle,therefore the distance
    represent the distance between the object's and the ego vehicle's boundary.
    :param  candidate_paths: The candidate paths
    :param objects: The objects around the ego vehicle, list of instances of the class Objects of package perception
    :param ego_vehicle: The ego vehicle
    """
    stop_vehicle = False
    max_safe_dist = 1.0
    min_safe_dist = 0.5
    min_radius = 40.0
    to_angle = 60.0 - ego_vehicle.speed * 3.6 if ego_vehicle.speed * 3.6 < 30.0 else 30.0
    from_angle = - to_angle
    t_react = 3.0
    radius = min_radius + ego_vehicle.speed * t_react
    ped_radius = 0.5
    ideal_dist = []
    objects, angles = objects_in_angle_range_and_in_radius(ego_vehicle, objects, from_angle, to_angle, radius)

    left_edge_dist, right_edge_dist = 0.0, 0.0
    for path in candidate_paths:
        flag_break = False
        d_min = float("inf")
        for p_x, p_y, p_yaw in zip(path.x, path.y, path.yaw):
            for object_i in objects:
                relative_d = math.hypot(object_i.y - p_y, object_i.x - p_x)
                relative_yaw = math.degrees(math.atan2(object_i.y - p_y, object_i.x - p_x))
                relative_yaw = 360.0+relative_yaw if relative_yaw < 0.0 else relative_yaw  # normalized in interval [0, 360)
                f1_p = relative_yaw - p_yaw
                f1_p = f1_p % 360.0  # normalized in interval [0, 360)
                f2_obj = (180.0-relative_yaw)
                f2_obj = 360.0 - f2_obj if f2_obj > 0.0 else -f2_obj
                f2_obj = f2_obj - object_i.yaw
                f2_obj = f2_obj % 360.0  # normalized in interval [0, 360)
                # Anticlockwise
                f1_p = 360.0 - f1_p
                f2_obj = 360.0 - f2_obj
                r1 = vehicle_boundary_to_angle(ego_vehicle.dimensions, f1_p)
                r2 = ped_radius if object_i.object_type == "pedestrian" else vehicle_boundary_to_angle(object_i.dimensions, f2_obj)
                d = relative_d - (r1+r2)  # distance between borders
                if d <= 0.0:
                    d_min = 0.0
                    flag_break = True
                    break
                else:
                    d_min = d if d_min > d else d_min
            if flag_break:
                break
        ideal_dist.append(d_min)

    min_dist = min(ideal_dist)
    max_dist = max(ideal_dist)+0.00001
    for i, dist in enumerate(ideal_dist):
        if dist > max_safe_dist:
            ideal_dist[i] = max_dist
        elif dist < min_safe_dist:
            ideal_dist[i] = min_dist
    """if all(ele == ideal_dist[0] for ele in ideal_dist):
        if left_edge_dist > right_edge_dist:
            ideal_dist[0] = max_dist
        else:
            ideal_dist[-2] = max_dist"""
    k = [(d_i-min_dist)/(max_dist-min_dist) for d_i in ideal_dist]
    # Check if there is no path without going over an obstacle and then stop the vehicle
    for i, object_i in enumerate(objects):
        rel_distance = math.hypot(object_i.y - ego_vehicle.y, object_i.x - ego_vehicle.x)
        angle = abs(angles[i])
        if object_i.object_type == "vehicle":
            if (angle < 20.0 and rel_distance < 6.0) or (angle < 10.0 and rel_distance < 7.0) or \
                    (((angle < 50.0 and rel_distance < 10) or (angle < 40.0 and rel_distance < 9.0) or
                      (angle < 30.0 and rel_distance < 10) or (angle < 20.0 and rel_distance < 12.0)) and k is None):
                stop_vehicle = True
        else:
            if (angle < 10.0 and rel_distance < 10.0) or (angle < 5.0 and rel_distance < 12.0):
                stop_vehicle = True
    return k, stop_vehicle


def objects_in_angle_range_and_in_radius(ego_vehicle, objects_list, from_angle=-90.0, to_angle=90.0, radius=20.0):
    """
    Method to find all the objects of a type like vehicles, pedestrians,etc
    between two angles (from_angle -> to_angle) in relation to vehicle coordinate system
    :param ego_vehicle: The self driving vehicle
    :param objects_list: The object list with vehicles and pedestrians
    :param from_angle: Start angle in relation to vehicle coordinate system in degrees in the interval [-180, 180)
    :param to_angle: The final angle in relation to vehicle coordinate system in degrees in the interval [-180, 180)
    :param radius: The max radius in which the object need to be
    """
    if len(objects_list) == 0:
        return [], []
    target_objects = []
    angle_list = []
    for an_object in objects_list:
        x = an_object.x - ego_vehicle.x
        y = an_object.y - ego_vehicle.y
        theta = math.degrees(math.atan2(y, x)) % 360.0
        theta = theta - ego_vehicle.yaw
        theta = theta % 360.0
        theta = theta - 360.0 if theta > 180.0 else theta
        rel_dist = math.hypot(an_object.y - ego_vehicle.y, an_object.x - ego_vehicle.x)
        if from_angle <= theta <= to_angle and rel_dist < radius:
            target_objects.append(an_object)
            # theta = theta + 360 if theta < 0 else theta
            angle_list.append(theta)
    return [object_i for object_i in target_objects] if len(target_objects) != 0 else [], angle_list


def distance_from_lane_borders(candidate_paths, left_border, right_border):
    """
    Method that calculate the closest distance from the points of the path to the boarders.
    :param  candidate_paths: The candidate paths
    :param left_border: The instance of class Boarder that represent the left boarder of a driving lane
    :param right_border: The instance of class Boarder that represent the right boarder of a driving lane
    """
    dist_from_border = []
    for path in candidate_paths:
        dist_min = float("inf")
        for b_x, b_y, b_lane_change in zip(left_border.x, left_border.y, left_border.lane_change):
            dx = path.x[-1] - b_x
            dy = path.y[-1] - b_y
            d = math.hypot(dx, dy)
            if dist_min > d:
                dist_min = d
        dist_from_border.append(dist_min)
    # The bigger the distance the better, so increase the cost for small distance
    inverse_distance = [1/d for d in dist_from_border]
    d_max = max(inverse_distance)
    d_min = min(inverse_distance)
    if d_max == 0.0:
        d_max = 0.001
    inverse_distance = [(d-d_min)/d_max for d in inverse_distance]  # Normalization in interval [0,1]

    return [float(1.0-d) for d in inverse_distance]






if __name__ == '__main__':
    pass
