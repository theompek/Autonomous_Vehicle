#!/usr/bin/env python

"""
This module use the carla simulation to collect data and train a HMM computing the calculate the computation matrices .

"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import math
import numpy as np
import itertools
from hmmlearn import hmm
import os.path

# from Perception import local_map as lmp

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================
STATE_VECTOR = ["Stop", "Decelerate", "Steady", "Accelerate"]
CRITERIA = ["DS", "U", "DV"]
CRITERIA_STATES = ["S", "B"]
OBSERVATION_VECTOR = [subset for subset in itertools.product(CRITERIA_STATES, repeat=len(CRITERIA))]
for case_i in OBSERVATION_VECTOR:
    temp = ""
    for s_i in case_i:
        temp += s_i
    OBSERVATION_VECTOR[OBSERVATION_VECTOR.index(case_i)] = temp
#print(OBSERVATION_VECTOR)
prev_state = STATE_VECTOR[2]

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

    def get_observation(self, vehicle, traject_with_constr, ego_vehicle=None):
        threshold_DS = 5
        threshold_U = 5.5
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

    def predict_state(self, vehicle_speed, obs_sequence):
        e = 0.1
        e_stop = 1
        start_probabilities = [0.25, 0.25, 0.25, 0.25]
        global prev_state
        current_state = STATE_VECTOR[2]
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
            #current_state = prev_state
            start_probabilities = [0.1, 0.1, 0.1, 0.1]
            start_probabilities[STATE_VECTOR.index(current_state)] = 0.7

        self.previous_speed = vehicle_speed
        self.HMM_model.startprob_ = np.array(start_probabilities)
        obs_seq_index = [OBSERVATION_VECTOR.index(ob) for ob in obs_sequence]
        _, seq = self.HMM_model.decode(np.array([obs_seq_index]).transpose())
        logprob = self.HMM_model.score(np.array([obs_seq_index]).transpose())
        prev_state = STATE_VECTOR[seq[-1]]
        return logprob, seq, current_state


def main():
    pass


if __name__ == '__main__':
    main()















