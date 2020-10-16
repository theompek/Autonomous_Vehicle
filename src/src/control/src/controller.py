#!/usr/bin/env python
from __future__ import print_function
"""
The module corresponds to the low level control system of the vehicle, contains the path following algorithms so at to the
vehicle follows the determined path and the algorithms for the speed control
"""

# ==============================================================================
# --General imports ------------------------------------------------------------
# ==============================================================================

from time import sleep
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
import time
from scipy.signal import butter, lfilter, freqz
import random

# ==============================================================================
# -- ROS imports ---------------------------------------------------------------
# ==============================================================================
import rospy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from control.msg import VehicleCmd
from perception.msg import Object
from perception.srv import EgoVehicleGeometry
from local_path_planner.msg import LocalOptimalPath
from std_msgs.msg import Float64

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================
# Global variables
LOOK_AHEAD = 6  # Pure pursuit follow the point that is LOOK_AHEAD points ahead the vehicle


def point_in_distance(x, y, dist=3.0):
    x_c, y_c = x[0], y[0]
    for i in range(1, len(x)):
        if np.hypot(x_c-x[i], y_c-y[i]) > dist:
            return i-1

    return len(x)-1


def calc_closer_point(x_base, y_base, x_v, y_v, x0=0, offset=1000000, step=1):
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

    return index


def pure_pursuit(tx, ty, vx, vy, v_yaw, L):
    """
    :param L: The length between front and rear wheels axis.
    :param v_yaw: The direction of the vehicle related to global coordinates
    :param vy: Vehicle y global position
    :param vx: Vehicle x global position
    :type ty: Target point y global position
    :param tx: Target point x global position

    """
    # Calculate pure pursuit
    a = math.atan2(ty - vy, tx - vx) - math.radians(v_yaw)
    ld = math.sqrt((ty - vy) ** 2 + (tx - vx) ** 2)
    delta = np.arctan(2 * L * math.sin(a) / ld)

    return delta


class SpeedController:
    def __init__(self):
        self.previous_e = 0.0
        self.previous_throttle = 0.0
        self.previous_brake = 0.0
        self.integral_time = 100
        self.saved_errors = [0.0]*self.integral_time
        self.Kp = 0.0025
        self.Kd = 2.2
        self.Ki = 0.00003
        self.Ke = 0.00095
        self.Kbr1 = 0.000035
        self.Kbr2 = 0.00075
        self.prev_time = time.time()
        self.time_step = 0.1

    def velocity_control(self, target_speed, current_speed):
        if time.time() - self.prev_time < self.time_step:
            self.prev_time = time.time()
            return self.previous_throttle, self.previous_brake
        # PD controller
        # Proportional
        e = target_speed - current_speed
        self.saved_errors.pop(0)
        self.saved_errors.append(e)
        integral = sum(self.saved_errors)
        # Derivative
        de = e - self.previous_e
        self.previous_e = e
        Uk = self.Ke*(self.Kp*e + self.Kd*de + self.Ki*integral)
        if 0.0 < abs(e) < 2.0:
            e = 0.0
        v_con_thr = self.previous_throttle + abs(e)*Uk
        v_con_br = 0.0
        if e < 0.0:
            v_con_thr = self.previous_throttle - abs(e*Uk)
            v_con_br = self.previous_brake + self.Kbr1*abs(e*Uk)*current_speed**2
        if target_speed < 3.0 and e < 0:
            v_con_thr = 0.0
            v_con_br = self.previous_brake + self.Kbr2*abs(e*Uk)*current_speed**2
        if target_speed == 0.0 and current_speed < 1:
            v_con_thr = 0.0
            v_con_br = 1

        v_con_thr = min(max(v_con_thr, 0.0), 1.0)
        v_con_br = min(max(v_con_br, 0.0), 1.0)
        self.previous_throttle = v_con_thr
        self.previous_brake = v_con_br
        return v_con_thr, v_con_br


class LP_Filter:
    def __init__(self):
        # Filter requirements.
        self.prev_time = time.time()
        self.data_length_throttle = 55
        self.data_length_brake = 10
        self.order = 2
        self.fs = 1200.0  # sample rate, Hz
        self.cutoff = 2.0  # desired cutoff frequency of the filter, Hz

    def butter_low_pass(self):
        self.fs = (self.fs + 1/(time.time()-self.prev_time+0.000001))/2
        self.cutoff = self.fs / 10
        self.prev_time = time.time()
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_low_pass_filter(self, data):
        b, a = self.butter_low_pass()
        y = lfilter(b, a, data)
        # print(self.fs)
        return min(max(0, y[-1]), 1)


class Control:
    def __init__(self):
        # --- General---
        self.file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_info_save.txt")
        with open(self.file_path, "a+") as file:
            file.truncate()
        self.steer_angle = 0.0
        self.veh_throttle = 0.0
        self.veh_brake = 0.0
        self.speed_controller = SpeedController()
        self.lp_filter = LP_Filter()
        self.throttle_data = [0]*self.lp_filter.data_length_throttle
        self.brake_data = [0]*self.lp_filter.data_length_brake
        self.last_simulator_time = 0.0
        self.current_simulator_time = 0.0
        # Save info for plots
        self.prev_time = time.time()
        self.time_step = 0.1
        self.dist_step = 0.1
        self.prev_veh_pos = [0, 0]

        # Distance between front and rear wheels and max steering angle
        self.L, self.max_angle = 2.0, 70.0
        # --- ROS ---
        rospy.init_node('Controller_node', anonymous=True)
        self.ego_vehicle = Object()
        self.optimal_local_path = LocalOptimalPath()
        self.subscriber_ego_vehicle = rospy.Subscriber("ego_vehicle_msg", Object, self.callback_ego_vehicle, queue_size=1)
        self.subscriber_optimal_local_path = rospy.Subscriber("optimal_local_path_msg", LocalOptimalPath,
                                                       self.callback_optimal_local_path, queue_size=1)
        self.subscriber_simulator_time_instance = rospy.Subscriber("simulator_time_instance_msg", Float64,
                                                                   self.callback_simulator_time_instance, queue_size=1)
        self.last_simulator_time = self.current_simulator_time
        self.client_ego_vehicle_geometry()
        self.pub_vehicle_controller = rospy.Publisher('vehicle_controller_msg', VehicleCmd, queue_size=1)

        while len(self.optimal_local_path.x) == 0 and not rospy.is_shutdown():
            print("Control node here, please wait the local path planning system .....")
            sleep(0.2)

    def control_vehicle(self):
        optimal_local_path = copy.deepcopy(self.optimal_local_path)
        emergency_stop = optimal_local_path.emergency_stop
        #print("emergency_stop", emergency_stop)
        ego_vehicle = copy.deepcopy(self.ego_vehicle)
        target_speed_id = optimal_local_path.direct_target_speed * 3.6 if optimal_local_path.direct_control else optimal_local_path.s_d[-1] * 3.6
        speed_look_ahead = int(target_speed_id / 10.0)
        id_point = calc_closer_point(optimal_local_path.x, optimal_local_path.y, ego_vehicle.x, ego_vehicle.y) + LOOK_AHEAD + speed_look_ahead
        id_point = len(optimal_local_path.x)-1 if id_point >= len(optimal_local_path.x) else id_point
        if id_point <= 0:
            self.veh_throttle = 0.0
            self.veh_brake = 1.0
            self.steer_angle = 0.0
            self.publish_vehicle_control()
            return True
        # Calculate pure pursuit steering angle
        delta = pure_pursuit(optimal_local_path.x[id_point], optimal_local_path.y[id_point], ego_vehicle.x, ego_vehicle.y, ego_vehicle.yaw, self.L)
        steer_angle = np.degrees(delta)
        steer_angle = self.max_angle if steer_angle > self.max_angle else steer_angle
        steer_angle = -self.max_angle if steer_angle < -self.max_angle else steer_angle
        self.steer_angle = steer_angle / self.max_angle
        # Calculate throttle and brake for velocity keeping with PID controller
        target_speed = optimal_local_path.direct_target_speed*3.6 if optimal_local_path.direct_control else optimal_local_path.s_d[id_point] * 3.6  # m/s to Km/s
        current_speed = ego_vehicle.speed * 3.6
        self.veh_throttle, self.veh_brake = self.speed_controller.velocity_control(target_speed, current_speed)
        self.throttle_data = self.throttle_data[1:] + [self.veh_throttle]
        self.veh_throttle = self.lp_filter.butter_low_pass_filter(self.throttle_data)
        self.throttle_data[-1] = self.veh_throttle
        #self.throttle_data = self.throttle_data[1:] + [self.veh_throttle]
        self.brake_data = self.brake_data[1:] + [self.veh_brake]
        self.veh_brake = self.lp_filter.butter_low_pass_filter(self.brake_data)
        #self.brake_data = self.brake_data[1:] + [self.veh_brake]
        if emergency_stop or (target_speed == 0.0 and current_speed < 2):
            self.veh_throttle = 0.0
            self.veh_brake = 1.0
            self.steer_angle = 0.0
        if target_speed == 0.0 and current_speed < 8.0:
            self.steer_angle = 0.0
        self.publish_vehicle_control()
        # Save info for plots
        '''
        if time.time() - self.prev_time > self.time_step and False:
            self.prev_time = time.time()
            output = [str(ob_i) for ob_i in [round(target_speed, 2), round(current_speed, 2), round(self.veh_throttle, 2), round(self.veh_brake, 2)]]
            output_str = ""
            for i, st_i in enumerate(output):
                output_str += st_i + "," if i != len(output)-1 else st_i
            with open(self.file_path, "a+") as file:
                file.write(output_str + "\n")
        if math.hypot(self.prev_veh_pos[0]-ego_vehicle.x, self.prev_veh_pos[1]-ego_vehicle.y) > self.dist_step:
            self.prev_veh_pos = [ego_vehicle.x, ego_vehicle.y]
            # output = [str(ob_i) for ob_i in [round(target_speed, 2), round(current_speed, 2), round(self.veh_throttle, 2), round(self.veh_brake, 2)]]
            output = [str(ob_i) for ob_i in
                      [round(optimal_local_path.x[id_point], 2), round(optimal_local_path.y[id_point], 2),
                       round(ego_vehicle.x, 2), round(ego_vehicle.y, 2)]]
            output_str = ""
            for i, st_i in enumerate(output):
                output_str += st_i + "," if i != len(output) - 1 else st_i
            with open(self.file_path, "a+") as file:
                file.write(output_str + "\n")'''
        '''
        if draw_point_flag:
            self.local_map.draw_paths(paths=[[carla.Location(x=ego_vehicle.x, y=ego_vehicle.y)]], life_time=0.2,
                                      color=[250, 250, 0], same_color=True)
            self.local_map.draw_paths(paths=[[carla.Location(x=optimal_frenet_path.x[id_point], y=optimal_frenet_path.y[id_point])]]
                                      , life_time=0.2, color=[250, 250, 0], same_color=True)
        '''
        return True

    # -------- ROS functions ---------
    def callback_ego_vehicle(self, ros_data):
        self.ego_vehicle = ros_data
        #rospy.loginfo(ros_data)

    def callback_simulator_time_instance(self, ros_data):
        self.current_simulator_time = ros_data.data
        #rospy.loginfo(ros_data)

    def callback_optimal_local_path(self, ros_data):
        self.optimal_local_path = ros_data
        #rospy.loginfo(ros_data)

    def client_ego_vehicle_geometry(self):
        rospy.wait_for_service('ego_vehicle_geometry_srv')
        try:
            vehicle_geometry = rospy.ServiceProxy('ego_vehicle_geometry_srv', EgoVehicleGeometry)
            resp1 = vehicle_geometry(0)
            #rospy.loginfo(resp1)
            self.L, self.max_angle = resp1.L, resp1.max_angle
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def publish_vehicle_control(self):
        pub = self.pub_vehicle_controller
        vehicle_controller = VehicleCmd()
        vehicle_controller.steer_angle = self.steer_angle
        vehicle_controller.veh_throttle = self.veh_throttle
        vehicle_controller.veh_brake = self.veh_brake
        #rospy.loginfo(vehicle_controller)
        pub.publish(vehicle_controller)


'''
def client_apply_vehicle_control(throttle, steer_angle, brake):
    rospy.wait_for_service('vehicle_controller_srv')
    try:
        vehicle_controller = rospy.ServiceProxy('vehicle_controller_srv', VehicleController)
        control_values = VehicleControllerRequest()
        control_values.steer_angle = steer_angle
        control_values.veh_throttle = throttle
        control_values.veh_brake = brake
        resp1 = vehicle_controller(control_values)
        # rospy.loginfo(resp1)
        return resp1
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
 '''

if __name__ == '__main__':
    #sleep(1)
    controller = Control()
    try:
        while not rospy.is_shutdown():
            controller.control_vehicle()
    except rospy.ROSInterruptException:
        print(3)
        pass
