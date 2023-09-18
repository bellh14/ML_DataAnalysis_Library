import numpy as np
import pandas as pd
import tensorflow as tf


class HMM:

    def __init__(self, n_states, n_observations, n_features,
                 n_steps, n_hidden):
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_features = n_features
        self.n_steps = n_steps
        self.n_hidden = n_hidden

        self.initial_prob = tf.placeholder(tf.float32, [self.n_states])
        self.transition_prob = tf.placeholder(tf.float32, [self.n_states,
                                                           self.n_states])
        self.emission_prob = tf.placeholder(tf.float32, [self.n_states,
                                                         self.n_observations])
        self.observations = tf.placeholder(tf.float32, [self.n_steps,
                                                        self.n_features])

        self.forward_prob = None
        self.backward_prob = None
        self.posterior_prob = None
        self.viterbi_prob = None
        self.viterbi_path = None

    def get_emission_prob(self, observations):
        return tf.transpose(tf.gather_nd(tf.transpose(self.emission_prob),
                                         observations))

    def foward_init(self):
        self.forward_prob = tf.multiply(self.initial_prob,
                                        self.get_emission_prob(
                                            self.observations[0]))
        self.forward_prob = tf.reshape(self.forward_prob, [1, -1])

    def foward(self):
        transitions = tf.matmul(self.forward_prob,
                                tf.transpose(self.get_emission_prob(
                                    self.observations[1])))
        weighted_transitions = tf.multiply(transitions, self.transition_prob)
        self.forward_prob = tf.reduce_sum(weighted_transitions, 0)
        return tf.reshape(self.foward_prob, tf.shape(self.forward_prob))
