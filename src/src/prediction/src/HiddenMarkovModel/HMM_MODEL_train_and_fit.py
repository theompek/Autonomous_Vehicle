#!/usr/bin/env python

"""
This module use the carla simulation to collect data and train a HMM computing the calculate the computation matrices .

"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import time
from time import sleep
import math
import numpy as np
import itertools
import random
import atexit
from hmmlearn import hmm
import sys
import os.path
sys.path.append('../Perception')
from Perception import local_map as lmp

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================
STATE_VECTOR = ["Stop", "Decelerate", "Steady", "Accelerate"]
CRITERIA = ["DS", "U", "DV"]  # DS: Distance from stop, u:speed, DV: distance from vehicle
CRITERIA_STATES = ["S", "B"]
OBSERVATION_VECTOR = [subset for subset in itertools.product(CRITERIA_STATES, repeat=len(CRITERIA))]
for case_i in OBSERVATION_VECTOR:
    temp = ""
    for s_i in case_i:
        temp += s_i
    OBSERVATION_VECTOR[OBSERVATION_VECTOR.index(case_i)] = temp


class HMM_MODEL:

    def __init__(self, local_map=None):
        if local_map is not None:  # <-- For training and fitting process
            self.local_map = local_map
            self.vehicle_id = None
            self.state_matrix = np.zeros((len(STATE_VECTOR), len(STATE_VECTOR)))
            self.observation_matrix = np.zeros((len(OBSERVATION_VECTOR), len(STATE_VECTOR)))
            self.previous_state = STATE_VECTOR[0]
            self.previous_speed = self.local_map.get_ego_vehicle().speed
            self.observations_list = []
            self.states_list = []
            self.HMM_model = None
        else:  # <-- For prediction process
            state_file_parameters = os.path.dirname(os.path.realpath(__file__))+'/fit_state_matrix.txt'
            observation_file_parameters = os.path.dirname(os.path.realpath(__file__))+'/fit_observation_matrix.txt'
            assert os.path.isfile(state_file_parameters) or os.path.isfile(observation_file_parameters)
            state_matrix = np.loadtxt(state_file_parameters, dtype=float)
            observation_matrix = np.loadtxt(observation_file_parameters, dtype=float)
            self.HMM_model = hmm.MultinomialHMM(n_components=4)
            self.HMM_model.startprob_ = np.array([0.5, 0.1, 0.3, 0.1])
            self.HMM_model.transmat_ = state_matrix
            self.HMM_model.emissionprob_ = observation_matrix
            self.previous_speed = None

    def get_trajectory_info(self, vehicle):
        min_dist = 5
        trajectories = self.local_map.find_all_driving_paths(vehicle, 30, 2)[0]
        traffic_signs_loc = self.local_map.get_traffic_stop_and_lights_objects_location()
        traject_with_constr = []
        for signs_loc in traffic_signs_loc:
            for trajectory in trajectories:
                for t_loc in trajectory:
                    dist = math.hypot(signs_loc.x - t_loc.x, signs_loc.y - t_loc.y)
                    if dist < min_dist:
                        v_dist = math.hypot(signs_loc.x - trajectory[0].x, signs_loc.y - trajectory[0].y)
                        traject_with_constr.append([trajectories.index(trajectory), v_dist])
                        break
        return traject_with_constr

    def get_observation(self, vehicle, traject_with_constr, ego_vehicle=None):
        threshold_DS = 2
        threshold_U = 2
        threshold_DV = 8
        DS = min([vti[2] for vti in traject_with_constr])
        U = vehicle.speed
        if ego_vehicle is None:
            DV = math.hypot(vehicle.y - self.local_map.get_ego_vehicle().y, vehicle.x - self.local_map.get_ego_vehicle().x)
        else:
            DV = math.hypot(vehicle.y - ego_vehicle.y, vehicle.x - ego_vehicle.x)

        if DS > threshold_DS:
            ds = CRITERIA_STATES[1]
        else:
            ds = CRITERIA_STATES[0]
        if U > threshold_U:
            u = CRITERIA_STATES[1]
        else:
            u = CRITERIA_STATES[0]
        if DV > threshold_DV:
            dv = CRITERIA_STATES[1]
        else:
            dv = CRITERIA_STATES[0]
        return ds+u+dv

    def calculate_state(self, current_speed, acceleration):
        e = 0.5
        e_stop = 1
        #print("current_speed", current_speed)
        #print("self.previous_speed", self.previous_speed)
        #print("acceleration", acceleration)
        du = self.previous_speed - current_speed
        if current_speed < e_stop:
            current_state = STATE_VECTOR[0]  # Stop
        elif acceleration < e:
            current_state = STATE_VECTOR[2]  # Steady
        elif du < 0:
            current_state = STATE_VECTOR[3]  # Accelerate
        else:
            current_state = STATE_VECTOR[1]  # Decelerate
        self.previous_speed = current_speed
        return current_state

    def calculate_matrices_by_count_cases(self, current_state, observation):
        prev_i = STATE_VECTOR.index(self.previous_state)
        curr_i = STATE_VECTOR.index(current_state)
        self.state_matrix[prev_i, curr_i] += 1
        obs_i = OBSERVATION_VECTOR.index(observation)
        self.observation_matrix[obs_i, curr_i] += 1
        self.previous_state = current_state

    def get_samples_for_model(self):
        ego_vehicle = self.local_map.get_ego_vehicle()
        observed_vehicle = None
        to_angle = 180 - ego_vehicle.speed * 3.6 if ego_vehicle.speed * 3.6 < 30 else 90
        from_angle = - to_angle
        t_react = 3
        radius = 50 + ego_vehicle.speed * t_react
        vehicles, angles = self.local_map.objects_in_angle_range_and_in_radius("vehicle", from_angle, to_angle, radius)

        observation = None
        if self.vehicle_id is not None:
            for vehicle in vehicles:
                if vehicle.object_id == self.vehicle_id:
                    tr_info = self.get_trajectory_info(vehicle)
                    if len(tr_info) > 0:
                        observation = self.get_observation(vehicle, tr_info)
                        observed_vehicle = vehicle
                        break
                    else:
                        self.vehicle_id = None
                        observed_vehicle = None
                        break

        if observation is None:
            for vehicle in vehicles:
                tr_info = self.get_trajectory_info(vehicle)
                if len(tr_info) > 0:
                    self.previous_speed = vehicle.speed
                    self.previous_state = self.calculate_state(vehicle.speed, vehicle.acceleration)
                    self.vehicle_id = vehicle.object_id
                    break

        if observation is not None:
            current_state = self.calculate_state(observed_vehicle.speed, observed_vehicle.acceleration)
            print(observation)
            print(current_state)
            print(observation)
            self.calculate_matrices_by_count_cases(current_state, observation)

    def get_a_observation_sequence(self):
        ego_vehicle = self.local_map.get_ego_vehicle()
        observed_vehicle = None
        to_angle = 180 - ego_vehicle.speed * 3.6 if ego_vehicle.speed * 3.6 < 30 else 90
        from_angle = - to_angle
        t_react = 3
        radius = 50 + ego_vehicle.speed * t_react
        vehicles, angles = self.local_map.objects_in_angle_range_and_in_radius("vehicle", from_angle, to_angle, radius)

        observation = None
        if self.vehicle_id is not None:
            for vehicle in vehicles:
                if vehicle.object_id == self.vehicle_id:
                    tr_info = self.get_trajectory_info(vehicle)
                    if len(tr_info) > 0:
                        observation = self.get_observation(vehicle, tr_info)
                        observed_vehicle = vehicle
                        break
                    else:
                        self.vehicle_id = None
                        observed_vehicle = None
                        break

        if observation is None:
            for vehicle in vehicles:
                tr_info = self.get_trajectory_info(vehicle)
                if len(tr_info) > 0:
                    self.previous_speed = vehicle.speed
                    self.previous_state = self.calculate_state(vehicle.speed, vehicle.acceleration)
                    self.vehicle_id = vehicle.object_id
                    break

        if observation is not None:
            current_state = self.calculate_state(observed_vehicle.speed, observed_vehicle.acceleration)
            #print(observation)
            #print(current_state)
            #print(observation)
            self.observations_list.append(observation)
            self.states_list.append(current_state)

    def fit_model(self):
        if os.path.isfile('state_matrix.txt'):
            if self.HMM_model is None:
                state_matrix = np.loadtxt('state_matrix.txt', dtype=int)
                observation_matrix = np.loadtxt('observation_matrix.txt', dtype=int)
                state_matrix = state_matrix / np.sum(state_matrix, axis=1)[:, np.newaxis]
                observation_matrix = observation_matrix / np.sum(observation_matrix, axis=1)[:, np.newaxis]
                self.HMM_model = hmm.MultinomialHMM(n_components=4)
                self.HMM_model.startprob_ = np.array([0.5, 0.1, 0.3, 0.1])
                self.HMM_model.transmat_ = state_matrix
                self.HMM_model.emissionprob_ = np.transpose(observation_matrix)
            # =========================
            # Fit the model
            observations = [OBSERVATION_VECTOR.index(ob) for ob in self.observations_list]
            print(len(observations))
            if len(self.observations_list) > 70:
                print("List of observations")
                print(self.observations_list)
                print(self.states_list)
                self.HMM_model.fit(np.array([observations]).transpose())
                self.observations_list = []
                # atexit.register(exit_fit_save_arrays, self.HMM_model.transmat_, self.HMM_model.emissionprob_)
                print("Matrix fit model")
                print(self.HMM_model.transmat_)

    def predict_state(self, vehicle_speed, obs_sequence):
        e = 0.1
        e_stop = 1
        start_probabilities = [0.25, 0.25, 0.25, 0.25]
        #print(obs_sequence)
        if self.previous_speed is not None:
            du = self.previous_speed - vehicle_speed
            if vehicle_speed < e_stop:
                current_state = STATE_VECTOR[0]  # Stop
            elif abs(du) < e:
                current_state = STATE_VECTOR[2]  # Steady
            elif du < 0:
                current_state = STATE_VECTOR[3]  # Accelerate
            else:
                current_state = STATE_VECTOR[1]  # Decelerate
            start_probabilities = [0.1, 0.1, 0.1, 0.1]
            start_probabilities[STATE_VECTOR.index(current_state)] = 0.7

        self.previous_speed = vehicle_speed
        self.HMM_model.startprob_ = np.array(start_probabilities)
        obs_seq_index = [OBSERVATION_VECTOR.index(ob) for ob in obs_sequence]
        _, seq = self.HMM_model.decode(np.array([obs_seq_index]).transpose())
        logprob = self.HMM_model.score(np.array([obs_seq_index]).transpose())

        return logprob, seq



def exit_handler(world):
    """
    Method to do some stuff when the program ends. It used to set the server in asynchronous mode
    :param world:
    """
    settings = world.get_settings()
    settings.fixed_delta_seconds = None
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("Exit from program")


def exit_save_arrays(client, world, HMM):
    state_matrix, observation_matrix = HMM.state_matrix, HMM.observation_matrix
    np.savetxt('state_matrix.txt', state_matrix, fmt='%d')
    np.savetxt('observation_matrix.txt', observation_matrix, fmt='%d')
    actors_list = world.get_actors().filter("*vehicle*")
    print(actors_list[0].type_id)
    [client.apply_batch([carla.command.DestroyActor(actor)]) for actor in actors_list
     if str(actor.type_id) != "vehicle.tesla.model3"]


def exit_fit_save_arrays(state_matrix, observation_matrix):
    state_matrix, observation_matrix = state_matrix, observation_matrix
    np.savetxt('fit_state_matrix.txt', state_matrix, fmt='%f')
    np.savetxt('fit_observation_matrix.txt', observation_matrix, fmt='%f')


def main():
    # fit_flag == False to collect data, fit_flag == True to fit he model.
    fit_flag = False
    save_flag = False
    load_flag = False
    synchronous_mode = False
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # seconds
    world = client.get_world()
    if synchronous_mode:
        atexit.register(exit_handler, world)
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.04
        settings.synchronous_mode = True
        world.apply_settings(settings)
        world.tick()
    actors_list = world.get_actors()
    my_actor = actors_list.filter("vehicle.tesla.model3*")
    vehicle = my_actor[0]
    local_map = lmp.LocalMap(vehicle, 400)
    HMM = HMM_MODEL(local_map)

    if os.path.isfile('state_matrix.txt') and load_flag:
        HMM.state_matrix = np.loadtxt('state_matrix.txt', dtype=int)
        HMM.observation_matrix = np.loadtxt('observation_matrix.txt', dtype=int)
        print("Load Matrices")
        print(HMM.observation_matrix)

    margin_x = 223
    margin_y = 12
    while True:
        v_x = random.randrange(220, 232, 1)
        v_y = random.randrange(-10, -3, 1)
        v_loc = carla.Location(x=v_x, y=v_y, z=vehicle.get_location().z)
        v_vel = random.randrange(0, 50, 1)/10
        vehicle.set_location(v_loc)
        # w = local_map.map.get_waypoint(v_loc, lane_type=carla.LaneType.Driving)
        transform = carla.Transform()
        transform.location = v_loc
        transform.rotation.yaw = 90
        vehicle.set_transform(transform)
        vehicle.set_velocity(carla.Vector3D(0, 0, 0))

        # Spawn vehicles
        vehicle_bp = random.choice(world.get_blueprint_library().filter('*vehicle*'))
        vehicle_actor = None
        v_loc = carla.Location(x=205, y=6, z=1)
        w = local_map.map.get_waypoint(v_loc, lane_type=carla.LaneType.Driving)
        transform = carla.Transform()
        transform.location = v_loc
        transform.rotation.yaw = w.transform.rotation.yaw
        while vehicle_actor is None:
            vehicle_actor = world.spawn_actor(vehicle_bp, transform)
        sleep(0.1)
        vehicle_actor.set_autopilot()
        sleep(2)
        start_time = time.time()
        while True:
            vehicle.set_velocity(carla.Vector3D(0, v_vel, 0))
            sp_loc = vehicle_actor.get_location()
            if sp_loc.x > margin_x or sp_loc.y > margin_y or (time.time() - start_time) > 10:
                client.apply_batch([carla.command.DestroyActor(vehicle_actor)])
                if fit_flag:
                    HMM.fit_model()
                sleep(0.2)
                break
            if not fit_flag:
                HMM.get_samples_for_model()
            else:
                HMM.get_a_observation_sequence()
            if save_flag:
                atexit.register(exit_save_arrays, client, world, HMM)
            sleep(0.2)


if __name__ == '__main__':
    main()















