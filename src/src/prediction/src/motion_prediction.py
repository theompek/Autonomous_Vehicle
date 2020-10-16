#!/usr/bin/env python
"""

"""
# ==============================================================================
# --General imports ------------------------------------------------------------
# ==============================================================================

from time import sleep
import math
import numpy as np
import sys
import copy

sys.path.append('./HiddenMarkovModel')

from HiddenMarkovModel.HMM_MODEL import *

# ==============================================================================
# -- ROS imports ---------------------------------------------------------------
# ==============================================================================
import rospy
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from perception.msg import Object
from perception.msg import ObjectsList
from perception.msg import WaypointsList
from perception.msg import Waypoint
from perception.msg import TrajectoriesList
from perception.srv import DrivingPaths, DrivingPathsRequest, DrivingPathsResponse
from prediction.msg import VehiclesCollisionEvent
from prediction.msg import VehiclesCollisionEventList
from prediction.msg import PedestrianCollisionEvent
from prediction.msg import PedestrianCollisionEventList
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================

MAX_SEQ_LENGTH_MEMORY = 5
MAX_STATE_SEQ_LENGTH_MEMORY = 1
GLOBAL_DRAW = True
state_i = 0
class VehicleState:
    def __init__(self, vehicle, angle):
        self.vehicle = [vehicle]
        self.angles = [angle]
        self.vehicle_id = vehicle.object_id
        self.trajectories = []
        self.possible_trajectory_id = []
        self.state = []
        self.observations_seq = []

    def v_pop(self, index=0):
        if len(self.vehicle) > MAX_SEQ_LENGTH_MEMORY:
            self.angles.pop(index)
            self.vehicle.pop(index)

    def s_pop(self, index=0):
        if len(self.state) > MAX_STATE_SEQ_LENGTH_MEMORY:
            self.state.pop(index)

    def ob_pop(self, index=0):
        if len(self.observations_seq) > MAX_STATE_SEQ_LENGTH_MEMORY:
            self.observations_seq.pop(index)


class MotionPrediction:

    def __init__(self):
        self.vehicles_state_list = []
        self.vehicles_id = []
        self.trajectories_length = 40.0
        self.HMM_MODEL = HMM_MODEL()
        self.vehicles_collision_events_info = []
        self.pedestrians_collision_events_info = []
        # --- ROS ---
        rospy.init_node('Prediction_node', anonymous=True)
        self.ego_vehicle = Object()
        self.vehicles_list = []
        self.pedestrians_list = []
        self.traffic_signs_loc = []
        self.ego_trajectory = []
        self.subscriber_ego_vehicle = rospy.Subscriber("ego_vehicle_msg", Object, self.callback_ego_vehicle,
                                                       queue_size=1)
        self.subscriber_ego_trajectory = rospy.Subscriber("ego_trajectory_msg", WaypointsList, self.callback_ego_trajectory,
                                                       queue_size=1)
        self.subscriber_vehicles_list = rospy.Subscriber('vehicles_list_msg', ObjectsList, self.callback_vehicles_list,
                                                         queue_size=1)
        self.subscriber_pedestrians_list = rospy.Subscriber('pedestrians_list_msg', ObjectsList, self.callback_pedestrians_list,
                                                         queue_size=1)
        self.subscriber_signs_location = rospy.Subscriber("signs_location_msg", WaypointsList, self.callback_signs_location,
                                                       queue_size=1)
        self.pub_vehicles_collision_info = rospy.Publisher('vehicles_collision_info_msg', VehiclesCollisionEventList, queue_size=1)
        self.pub_pedestrians_collision_info = rospy.Publisher('pedestrians_collision_info_msg', PedestrianCollisionEventList, queue_size=1)
        # Draw trajectories
        if GLOBAL_DRAW:
            self.pub_prototype_trajectories_draw = rospy.Publisher('prototype_trajectories_draw_msg', TrajectoriesList, queue_size=1)
            self.pub_collision_points_draw = rospy.Publisher('collision_points_draw_msg', WaypointsList, queue_size=1)

    def track_vehicles_and_save_states(self, vehicles, angles):
        # For vehicles found
        for vehicle in vehicles:
            if vehicle.object_id in self.vehicles_id:
                index = self.vehicles_id.index(vehicle.object_id)
                veh_state = self.vehicles_state_list[index]
                veh_state.vehicle.append(vehicle)
                veh_state.angles.append(angles[vehicles.index(vehicle)])
                veh_state.v_pop(0)
            else:
                self.vehicles_state_list.append(VehicleState(vehicle, angles[vehicles.index(vehicle)]))
                self.vehicles_id.append(vehicle.object_id)

        # Remove vehicles from list which don't exist any more
        if not vehicles:
            self.vehicles_state_list = []
            self.vehicles_id = []
        existed_vehicles_id = [veh.object_id for veh in vehicles]
        #print(existed_vehicles_id)
        if len(existed_vehicles_id) != 0:
            self.vehicles_state_list = [self.vehicles_state_list[i] for i in range(len(self.vehicles_state_list)) if self.vehicles_id[i] in existed_vehicles_id]
            self.vehicles_id = [v_id for v_id in self.vehicles_id if v_id in existed_vehicles_id]

    def get_objects_around_vehicle(self, obj_type="vehicle", min_radius=50.0):
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        if obj_type == "vehicle":
            to_angle = 120.0 - ego_vehicle.speed * 3.6 if ego_vehicle.speed * 3.6 < 30.0 else 90.0
        else:
            to_angle = 80.0 - ego_vehicle.speed * 3.6 if ego_vehicle.speed * 3.6 < 30.0 else 40.0
        from_angle = - to_angle
        t_stop = 3.0
        radius = min_radius + ego_vehicle.speed * t_stop
        objects, angles = self.objects_in_angle_range_and_in_radius(obj_type, from_angle, to_angle, radius)
        return objects, angles

    def objects_in_angle_range_and_in_radius(self, object_type, from_angle=-90.0, to_angle=90.0, radius=20.0):
        """
        Method to find all the objects of a type like vehicles, pedestrians,etc
        between two angles (from_angle -> to_angle) in relation to vehicle coordinate system
        :param ego_vehicle: The self driving vehicle
        :param object_type: The object type, vehicles, pedestrians, traffic signs etc.
        :param from_angle: Start angle in relation to vehicle coordinate system in degrees in the interval [-180, 180)
        :param to_angle: The final angle in relation to vehicle coordinate system in degrees in the interval [-180, 180)
        :param radius: The max radius in which the object need to be
        """
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        if object_type == "vehicle":
            objects_list = copy.deepcopy(self.vehicles_list)
        elif object_type == "pedestrian":
            objects_list = copy.deepcopy(self.pedestrians_list)
        else:
            return [], []
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

    def get_prototype_trajectories_and_vehicles(self):
        draw_points_flag = True
        percentage_renew = 0.5
        vehicles, angles = self.get_objects_around_vehicle(obj_type="vehicle")
        self.track_vehicles_and_save_states(vehicles, angles)
        veh_trajectories = self.client_driving_paths(vehicles)
        if vehicles:
            for v_i, vehicle in enumerate(vehicles):
                index = self.vehicles_id.index(vehicle.object_id)
                if len(self.vehicles_state_list[index].trajectories) == 0:
                    self.vehicles_state_list[index].trajectories = veh_trajectories[v_i]
                else:
                    w_e = self.vehicles_state_list[index].trajectories[0][-1]
                    w_b = self.vehicles_state_list[index].trajectories[0][0]
                    dist = math.hypot(vehicle.x - w_e.x, vehicle.y - w_e.y)
                    dist_reverse = math.hypot(vehicle.x - w_b.x, vehicle.y - w_b.y)
                    if dist < percentage_renew*self.trajectories_length or \
                            dist_reverse > (percentage_renew/4)*self.trajectories_length or \
                            len(self.vehicles_state_list[index].trajectories) == 1:
                        self.vehicles_state_list[index].trajectories = veh_trajectories[v_i]
        if GLOBAL_DRAW:
            self.publish_prototype_trajectories_draw()
        return [state.vehicle[-1] for state in self.vehicles_state_list], [state.trajectories for state in self.vehicles_state_list]

    def calculate_trajectories_probability(self):
        """
        For each vehicle saved in data base we calculate the most possible trajectory from the prototypes trajectories
        to follow. For each vehicle track we take all the past instances of the vehicle and we calculate the minimum
        distance of each from each trajectory and sum them. Finally the trajectory with the lowest sum is chosen.
        The most possible trajectory and the position of the vehicle on it is saved in "possible_trajectory_id"
        """
        e_threshold = 0.1
        min_probability = 0.3
        max_probability = 0.6
        trajectories_list = []
        probabilities = []
        for track in self.vehicles_state_list:
            x_y = [[vehicle_inst.x, vehicle_inst.y] for vehicle_inst in track.vehicle]
            min_sums = []
            index = []
            yaw_cos = []
            for trajectory in track.trajectories:
                dist_min_list = []
                idx = 0
                for i in range(len(x_y)):
                    min_dist = float("inf")
                    for w in trajectory:
                        dist = math.hypot(w.x - x_y[i][0], w.y - x_y[i][1])
                        if min_dist > dist:
                            min_dist = dist
                            if i == len(x_y)-1:
                                idx = trajectory.index(w)
                    dist_min_list.append(min_dist)
                yaw_cos.append(sum([math.cos(math.radians(tr.yaw)) for tr in trajectory[idx:]])/len(trajectory[idx:]))
                index.append(idx)
                min_sums.append(sum(dist_min_list))

            k1 = abs(math.cos(math.radians(track.vehicle[-1].yaw)))
            max_yaw = [1-abs(k1 - abs(c_yaw)) for c_yaw in yaw_cos]
            max_sums = [1-m_s/(max(min_sums)+0.0000001) for m_s in min_sums]
            max_value = [(max_sums[i]+max_yaw[i]**2) for i in range(len(max_sums))]
            probability = [m_v/(sum(max_value)+0.000001) for m_v in max_value]
            max_of_all = max(probability)
            track.possible_trajectory_id = []
            for i in range(len(probability)):
                if abs(max_of_all-probability[i]) < e_threshold or \
                        probability[probability.index(max_of_all)] < min_probability or\
                        probability[i] > max_probability:
                    track.possible_trajectory_id.append([i, index[i]])
            if len(track.possible_trajectory_id) == 0:
                i = probability.index(max_of_all)
                track.possible_trajectory_id.append([i, index[i]])
            trajectories = []
            for i in range(len(track.trajectories)):
                pos_tr = [p_t[0] for p_t in track.possible_trajectory_id]
                if i in pos_tr:
                    trajectories.append(track.trajectories[i][track.possible_trajectory_id[pos_tr.index(i)][1]:])
            track.trajectories = trajectories
            probabilities.append(probability)
            trajectories_list.append(trajectories)
        return trajectories_list, probabilities

    def get_trajectories_with_stop_constraints(self):
        min_dist = 3.0
        vehicles = [track.vehicle[-1] for track in self.vehicles_state_list]
        trajectories = [track.trajectories for track in self.vehicles_state_list]
        traffic_signs_loc = copy.deepcopy(self.traffic_signs_loc)
        traject_with_constr = []
        for signs_loc in traffic_signs_loc:
            for v_t in trajectories:
                for trajectory in v_t:
                    for t_loc in trajectory:
                        dist = math.hypot(signs_loc.x - t_loc.x, signs_loc.y - t_loc.y)
                        if dist < min_dist:
                            vehicle = vehicles[trajectories.index(v_t)]
                            v_dist = math.hypot(signs_loc.x - vehicle.x, signs_loc.y - vehicle.y)
                            traject_with_constr.append([trajectories.index(v_t), v_t.index(trajectory), v_dist])
                            break
        return traject_with_constr

    def predict_vehicle_speed(self, traject_with_constr):
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        vehicles_speed = []
        speed_probabilities = []
        tick_time = 1.0
        a_stop = 6.0
        a_deceleration = 4.0
        a_acceleration = 3.0
        global state_i
        for v_i in range(len(self.vehicles_state_list)):
            vehicle = self.vehicles_state_list[v_i].vehicle[-1]
            traf_info_const = [const_info for const_info in traject_with_constr if v_i == const_info[0]]
            if len(traf_info_const) != 0:
                observation = self.HMM_MODEL.get_observation(vehicle, traf_info_const, ego_vehicle)
                #print(observation)
                self.vehicles_state_list[v_i].observations_seq.append(observation)
                self.vehicles_state_list[v_i].ob_pop(0)
                #print(self.vehicles_state_list[v_i].observations_seq)
                ln_prob, num_seq, curr_state = self.HMM_MODEL.predict_state(vehicle.speed, self.vehicles_state_list[v_i].observations_seq)
                probability = math.exp(ln_prob)
                states = [STATE_VECTOR[i] for i in num_seq]
                #print(states)
                next_state = states[-1]
                state_i = OBSERVATION_VECTOR.index(observation)
            else:
                probability = 1
                next_state = STATE_VECTOR[2]  # Steady state
                curr_state = next_state

            if next_state == STATE_VECTOR[2]:  # Steady state
                predicted_speed = vehicle.speed
            elif next_state == STATE_VECTOR[0]:  # Stop
                predicted_speed = vehicle.speed - tick_time*a_stop
                predicted_speed = 0.0 if predicted_speed < 0.0 else predicted_speed
            elif next_state == STATE_VECTOR[3]:  # Accelerate
                predicted_speed = vehicle.speed + tick_time * a_acceleration*probability
            else:  # Deceleration
                predicted_speed = vehicle.speed - tick_time * a_deceleration*probability
                predicted_speed = 0.0 if predicted_speed < 0.0 else predicted_speed
            speed_probabilities.append(probability)
            vehicles_speed.append(predicted_speed)
            #print(STATE_VECTOR.index(next_state), state_i, round(predicted_speed, 2), round(vehicle.speed, 2), round(probability, 2), STATE_VECTOR.index(curr_state))
        return vehicles_speed, speed_probabilities

    def predict_vehicles_collision(self):
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        draw_points_flag = True
        ego_vehicle_trajectory = copy.deepcopy(self.ego_trajectory)
        if len(ego_vehicle_trajectory) == 0:
            return True
        vehicles, all_vehicles_paths = self.get_prototype_trajectories_and_vehicles()
        possible_trajectories, trajectories_probabilities = self.calculate_trajectories_probability()
        traject_with_constr = self.get_trajectories_with_stop_constraints()
        predicted_vehicles_speed, speed_probabilities = self.predict_vehicle_speed(traject_with_constr)
        # angles = [angles[i] for i in range(len(vehicles)) if vehicles[i] in c_vehicles]
        min_collision_dist = 2.0
        t = [0.0]
        ego_vehicle_speed = ego_vehicle.speed + 0.00000001
        dist = 0.0
        for i_d in range(len(ego_vehicle_trajectory)-1):
            dist += math.hypot(ego_vehicle_trajectory[i_d + 1].y - ego_vehicle_trajectory[i_d].y,
                            ego_vehicle_trajectory[i_d + 1].x - ego_vehicle_trajectory[i_d].x)
            t.append(dist/ego_vehicle_speed if ego_vehicle_speed > 0.0 else 100000.0)

        t_v = []
        for trajectories in possible_trajectories:
            t_tr = []
            other_vehicles_speed = predicted_vehicles_speed[possible_trajectories.index(trajectories)] + 0.001
            for trajectory in trajectories:
                t_t = [0.0]
                dist = 0.0
                for i_d in range(len(trajectory) - 1):
                    dist += math.hypot(trajectory[i_d + 1].y - trajectory[i_d].y,
                                       trajectory[i_d + 1].x - trajectory[i_d].x)
                    t_t.append(dist / other_vehicles_speed if other_vehicles_speed > 0.0 else 100000.0)
                t_tr.append(t_t)
            t_v.append(t_tr)

        collision_vehicles = []
        for t_tr in t_v:
            #  t_windows = min_collision_dist/vehicles[t_v.index(t_tr)].speed + 0.001
            for t_t in t_tr:
                break_flag = False
                for time_i in t_t:
                    bigger_than = False
                    for time_j in t:
                        if time_j >= time_i:
                            bigger_than = True
                            w1 = possible_trajectories[t_v.index(t_tr)][t_tr.index(t_t)][t_t.index(time_i)]
                            w2 = ego_vehicle_trajectory[t.index(time_j)]
                            dist = math.hypot(w2.y - w1.y, w2.x - w1.x)
                            if dist < min_collision_dist:
                                break_flag = True
                                collision_time = time_i
                                collision_vehicles.append([t_v.index(t_tr), t_tr.index(t_t), collision_time, w1])
                            break
                    if not bigger_than and time_i != 0.0:
                        for time_j in t:
                            w1 = possible_trajectories[t_v.index(t_tr)][t_tr.index(t_t)][t_t.index(time_i)]
                            w2 = ego_vehicle_trajectory[t.index(time_j)]
                            dist = math.hypot(w2.y - w1.y, w2.x - w1.x)
                            if dist < min_collision_dist:
                                break_flag = True
                                collision_time = time_i
                                if math.hypot(w2.y - ego_vehicle_trajectory[0].y, w2.x - ego_vehicle_trajectory[0].x) > \
                                        math.hypot(w2.y - possible_trajectories[t_v.index(t_tr)][t_tr.index(t_t)][0].y,
                                                   w2.x - possible_trajectories[t_v.index(t_tr)][t_tr.index(t_t)][0].x):
                                    collision_time = time_j
                                collision_vehicles.append([t_v.index(t_tr), t_tr.index(t_t), collision_time, w1])
                                break
                    if break_flag:
                        break

        collision_events_info = []
        for collision in collision_vehicles:
            vehicle = self.vehicles_state_list[collision[0]].vehicle[-1]
            trajectory_probability = trajectories_probabilities[collision[0]][collision[1]]
            speed_probability = speed_probabilities[collision[0]]
            collision_time = collision[2]
            collision_point = collision[3]
            predict_probability = trajectory_probability*speed_probability
            collision_events_info.append([vehicle, collision_point, collision_time, predict_probability])
        self.vehicles_collision_events_info = collision_events_info
        self.publish_vehicles_collision_events()
        if GLOBAL_DRAW and len(vehicles) != 0 and len(collision_vehicles) != 0:
            self.publish_collision_points_draw([collision[-1] for collision in collision_vehicles])

    def predict_pedestrians_collision(self):
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        radius_sum = 2.0
        pedestrians, angles = self.get_objects_around_vehicle(obj_type="pedestrian", min_radius=12)
        t_col = []
        dist_col = []
        for pedestrian in pedestrians:
            t = 100000.0
            min_dist = 10000.0
            relative_speed_x = pedestrian.vel_x - ego_vehicle.vel_x - math.cos(math.radians(ego_vehicle.yaw))*0.1
            relative_speed_y = pedestrian.vel_y - ego_vehicle.vel_y - math.sin(math.radians(ego_vehicle.yaw))*0.1
            relative_distance_x = pedestrian.x - ego_vehicle.x
            relative_distance_y = pedestrian.y - ego_vehicle.y
            a = relative_speed_x**2 + relative_speed_y**2 + 0.000001
            b = 2*(relative_speed_x*relative_distance_x + relative_speed_y*relative_distance_y)
            c = relative_distance_x**2 + relative_distance_y**2 - radius_sum**2
            discriminant = b**2 - 4*a*c
            if discriminant > 0.0:  # Collision happens
                t0 = (-b - math.sqrt(discriminant)) / (2.0 * a)
                t1 = (-b + math.sqrt(discriminant)) / (2.0 * a)
                t = min([t0, t1]) if min([t0, t1]) >= 0.0 else max([t0, t1])
                min_dist = 0.0
            else:
                t = -b/(2*a)
                min_dist = a * (t ** 2) + t * b + c

            t, min_dist = (100000.0, 10000.0) if t < 0.0 else (t, min_dist)  # No collision if t < 0
            t_col.append(t)
            dist_col.append(min_dist)

        if t_col:
            collision_event = [pedestrians[dist_col.index(min(dist_col))], angles[dist_col.index(min(dist_col))],
                               t_col[dist_col.index(min(dist_col))], min(dist_col)]
        else:
            collision_event = []
        self.pedestrians_collision_events_info = [collision_event]
        self.publish_pedestrians_collision_events()
        if GLOBAL_DRAW and len(collision_event) != 0 and collision_event[-1] < 2.0:
            pedestrian_object = collision_event[0]
            dist_ped = math.hypot(pedestrian_object.x-ego_vehicle.x, pedestrian_object.y-ego_vehicle.y)
            angle = math.radians(collision_event[1])
            dist_ped = dist_ped*math.cos(angle)
            waypoint = Waypoint()
            theta = math.radians(ego_vehicle.yaw)
            waypoint.x = ego_vehicle.x + dist_ped * math.cos(theta)
            waypoint.y = ego_vehicle.y + dist_ped * math.sin(theta)
            self.publish_collision_points_draw([waypoint])

    # -------- ROS functions ---------
    def callback_ego_vehicle(self, ros_data):
        self.ego_vehicle = ros_data

    def callback_vehicles_list(self, ros_data):
        self.vehicles_list = ros_data.objects_list

    def callback_pedestrians_list(self, ros_data):
        self.pedestrians_list = ros_data.objects_list

    def callback_signs_location(self, ros_data):
        self.traffic_signs_loc = ros_data.waypoints_list

    def callback_ego_trajectory(self, ros_data):
        self.ego_trajectory = ros_data.waypoints_list

    def client_driving_paths(self, vehicles):
        rospy.wait_for_service('driving_paths_srv')
        rospy.wait_for_service('driving_paths_srv')
        x_list = [vehicle.x for vehicle in vehicles]
        y_list = [vehicle.y for vehicle in vehicles]
        try:
            driving_paths = rospy.ServiceProxy('driving_paths_srv', DrivingPaths)
            resp1 = driving_paths(x_list, y_list, self.trajectories_length)
            veh_trajectories = []
            for vehicle in resp1.driving_paths:
                trajectories = [trajectory.waypoints_list for trajectory in vehicle.trajectories_list]
                veh_trajectories.append(trajectories)
            #rospy.loginfo(veh_trajectories)
            return veh_trajectories
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def publish_vehicles_collision_events(self):
        pub = self.pub_vehicles_collision_info
        collision_event_list = []
        for collision_event in self.vehicles_collision_events_info:
            ros_vehicle_collision_event = VehiclesCollisionEvent()
            ros_vehicle_collision_event.object = collision_event[0]
            ros_vehicle_collision_event.collision_point = collision_event[1]
            ros_vehicle_collision_event.collision_time = collision_event[2]
            ros_vehicle_collision_event.prediction_probability = collision_event[3]
            collision_event_list.append(ros_vehicle_collision_event)
        ros_collision_event_list = VehiclesCollisionEventList()
        ros_collision_event_list.collision_event_list = collision_event_list
        #rospy.loginfo(ros_collision_event_list)
        pub.publish(ros_collision_event_list)

    def publish_pedestrians_collision_events(self):
        pub = self.pub_pedestrians_collision_info
        collision_event_list = []
        if self.pedestrians_collision_events_info[0]:
            for collision_event in self.pedestrians_collision_events_info:
                ros_collision_event = PedestrianCollisionEvent()
                ros_collision_event.object = collision_event[0]
                ros_collision_event.angle = collision_event[1]
                ros_collision_event.collision_time = collision_event[2]
                ros_collision_event.collision_distance = collision_event[3]
                collision_event_list.append(ros_collision_event)
        ros_collision_event_list = PedestrianCollisionEventList()
        ros_collision_event_list.collision_event_list = collision_event_list
        # rospy.loginfo(ros_collision_event)
        pub.publish(ros_collision_event_list)

    def publish_prototype_trajectories_draw(self):
        pub = self.pub_prototype_trajectories_draw
        trajectories_list = []
        for prototype_trajectories in [veh.trajectories for veh in self.vehicles_state_list]:
            for trajectory in prototype_trajectories:
                ros_trajectory = WaypointsList()
                ros_trajectory.waypoints_list = trajectory
                trajectories_list.append(ros_trajectory)
        prototype_trajectories = TrajectoriesList()
        prototype_trajectories.trajectories_list = trajectories_list
        #rospy.loginfo(prototype_trajectories)
        pub.publish(prototype_trajectories)

    def publish_collision_points_draw(self, collision_points):
        pub = self.pub_collision_points_draw
        ros_points_list = WaypointsList()
        ros_points_list.waypoints_list = collision_points
        #rospy.loginfo(prototype_trajectories)
        pub.publish(ros_points_list)


def main():
    motion_prediction = MotionPrediction()
    try:
        while not rospy.is_shutdown():
            motion_prediction.predict_vehicles_collision()
            motion_prediction.predict_pedestrians_collision()
    except rospy.ROSInterruptException:
        print("Local path planner node failed")
        pass


if __name__ == '__main__':
    main()














