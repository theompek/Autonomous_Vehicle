#!/usr/bin/env python

"""
This module use implement a routine to calculate the probability values of the samples in matrices

"""
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import os.path
from Prediction.HiddenMarkovModel.HMM_MODEL import *

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================


def main():
    if os.path.isfile('state_matrix.txt'):
        state_matrix = np.loadtxt('state_matrix.txt', dtype=int)
        observation_matrix = np.loadtxt('observation_matrix.txt', dtype=int)
        print("Load state_matrix Matrices")
        print(state_matrix)
        print(np.sum(state_matrix, axis=1))
        state_matrix = state_matrix / np.sum(state_matrix, axis=1)[:,  np.newaxis]
        print(state_matrix)
        print(np.sum(state_matrix, axis=1))
        print("")
        print("Load observation_matrix Matrices")
        print(observation_matrix)
        print(np.sum(observation_matrix, axis=1))
        observation_matrix = observation_matrix / np.sum(observation_matrix, axis=1)[:, np.newaxis]
        print(observation_matrix)
        print(np.sum(observation_matrix, axis=1))
        # =========================

        start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
        transition_probability = {
            'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
            'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
        }
        emission_probability = {
            'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
            'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
        }

        model = hmm.MultinomialHMM(n_components=4)
        model.startprob_ = np.array([0.5, 0.1, 0.3, 0.1])
        model.transmat_ = state_matrix
        model.emissionprob_ = np.transpose(observation_matrix)
        # Use samples by hand
        print("samples by hand")
        obs_sequence = ["SBS", "SBS"]
        obs_seq_index = [OBSERVATION_VECTOR.index(ob) for ob in obs_sequence]
        logprob, seq = model.decode(np.array([obs_seq_index]).transpose())
        print(math.exp(logprob))
        print(seq)
        print([STATE_VECTOR[i] for i in seq])

        # Use sample from model
        print("samples from model")
        sample = model.sample(10)
        obs_seq_index = [i[0] for i in sample[0]]
        print(obs_seq_index)
        logprob, seq = model.decode(np.array([obs_seq_index]).transpose())
        print("predict", [STATE_VECTOR[i] for i in seq])
        print("actual", [STATE_VECTOR[i] for i in sample[1]])
        print("model_old")
        print(model.transmat_)
        #print(model.emissionprob_)

        # Train the model
        print("model_new")
        sample2 = model.sample(1000)
        obs_seq_index2 = [i[0] for i in sample2[0]]
        model_new = model.fit(np.array([obs_seq_index2]).transpose())
        print(model_new.transmat_)
        logprob, seq = model.decode(np.array([obs_seq_index]).transpose())
        print("predict", [STATE_VECTOR[i] for i in seq])
        print("actual", [STATE_VECTOR[i] for i in sample[1]])


def main2():
    if os.path.isfile('fit_state_matrix.txt'):
        state_matrix = np.loadtxt('state_matrix.txt', dtype=int)
        observation_matrix = np.loadtxt('observation_matrix.txt', dtype=int)
        state_matrix = state_matrix / np.sum(state_matrix, axis=1)[:,  np.newaxis]
        observation_matrix = observation_matrix / np.sum(observation_matrix, axis=1)[:, np.newaxis]
        model1 = hmm.MultinomialHMM(n_components=4)
        model1.startprob_ = np.array([0.5, 0.1, 0.3, 0.1])
        model1.transmat_ = state_matrix
        model1.emissionprob_ = np.transpose(observation_matrix)

        # After fitting
        state_matrix = np.loadtxt('fit_state_matrix.txt', dtype=float)
        observation_matrix = np.loadtxt('fit_observation_matrix.txt', dtype=float)
        model2 = hmm.MultinomialHMM(n_components=4)
        model2.startprob_ = np.array([0.5, 0.1, 0.3, 0.1])
        model2.transmat_ = state_matrix
        model2.emissionprob_ = observation_matrix

        # Use samples by hand
        print("samples by hand")
        obs_sequence = ['BBB', "SBB", "SSB", "SSB", "SSB", "BBB"]
        obs_seq_index = [OBSERVATION_VECTOR.index(ob) for ob in obs_sequence]
        # Model1
        print(model1.transmat_)
        logprob, seq = model1.decode(np.array([obs_seq_index]).transpose())
        print(math.exp(logprob))
        print(seq)
        print([STATE_VECTOR[i] for i in seq])
        # Model2
        print(model2.transmat_)
        logprob, seq = model2.decode(np.array([obs_seq_index]).transpose())
        print(math.exp(logprob))
        print(seq)
        print([STATE_VECTOR[i] for i in seq])



if __name__ == '__main__':
    main2()


