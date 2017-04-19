# !/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import operator
import os
from collections import defaultdict
from itertools import groupby

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn

from visits_detector.core.components.data_structures import EventType, Event
from visits_detector.core.components.event_type_determiner.event_type_determiner_base import EventTypeDeterminerBase

logging.basicConfig(level=logging.DEBUG)


class NNParams(object):
    train_keep_prob = 1.0
    n_epoch = 10
    grad_norm_clip = 15.0
    rnn_cell_size = 32
    learning_rate = 1e-4
    n_dim = 2


class BiDirBinaryDynamicLSTMPredictor(object):
    def __init__(self, nn_params, state_file_name, model_file_name):
        self.nn_params = nn_params
        self.state = self._load_state(state_file_name)
        self._restore_session(model_file_name)

    @staticmethod
    def _load_state(state_file_name):
        if os.path.exists(state_file_name):
            with open(state_file_name) as f:
                state = defaultdict(dict)

                for k, v in json.load(f).iteritems():
                    state[k] = v

                return state
        raise RuntimeError('State file not found')

    def _restore_session(self, model_file_name):
        train_mean = np.array(self.state['train_mean'])
        train_std = np.array(self.state['train_std'])
        track_len_mean, track_len_std = self.state['track_len_mean'], self.state['track_len_std']
        points_within_50m_mean_count, points_within_50m_std_count = self.state['points_within_50m_mean_count'], \
                                                                    self.state['points_within_50m_std_count']

        self._define_graph(train_mean, train_std, track_len_mean, track_len_std,
                           points_within_50m_mean_count, points_within_50m_std_count)
        self.session_ = tf.Session(graph=self.graph_)
        self.saver_.restore(self.session_, model_file_name)

    def _get_optimizer(self, loss, lrate):
        optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
        gradvars = optimizer.compute_gradients(loss)
        gradients, v = zip(*gradvars)
        gradients, _ = tf.clip_by_global_norm(gradients, self.nn_params.grad_norm_clip)
        return optimizer.apply_gradients(zip(gradients, v))

    def _define_graph(self, train_mean, train_std, track_len_mean, track_len_std, points_within_50m_mean_count,
                      points_within_50m_std_count):
        self.graph_ = tf.Graph()

        with self.graph_.as_default():
            self.input_track_ = tf.placeholder(tf.float32,
                                               shape=(None, 1, self.nn_params.n_dim))  # time x batch x feats
            self.input_track_len_ = tf.placeholder(tf.int32, shape=(1, 1))
            self.target_class_ = tf.placeholder(tf.int32, shape=(1, 1))  # batch x feats
            self.points_within_50m_count_ = tf.placeholder(tf.int32, [1])
            # batch is set to 1 here, this is not computationally efficient
            # for batch_size > 1 you have to take care of padding and masking, so leaving this out for now

            learning_rate = tf.placeholder_with_default(input=self.nn_params.learning_rate, shape=())
            self.track_len_ = tf.placeholder(tf.int32, [1])
            self.keep_prob_ = tf.placeholder_with_default(input=1.0, shape=())

            normalized_inputs = (self.input_track_ - train_mean) / train_std
            # normalized_inputs = tf.nn.dropout(x=normalized_inputs, keep_prob=self.keep_prob_)

            normalized_tanh = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True,
                                                                     activation_fn=tf.nn.tanh, trainable=True)
            fw_rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.nn_params.rnn_cell_size,
                                                  num_proj=self.nn_params.rnn_cell_size,
                                                  state_is_tuple=True,
                                                  activation=normalized_tanh)
            bw_rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.nn_params.rnn_cell_size,
                                                  num_proj=self.nn_params.rnn_cell_size,
                                                  state_is_tuple=True,
                                                  activation=normalized_tanh)

            outputs, final_states = rnn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                                  bw_rnn_cell,
                                                                  inputs=normalized_inputs,
                                                                  sequence_length=self.track_len_,
                                                                  initial_state_fw=None,
                                                                  initial_state_bw=None,
                                                                  dtype=tf.float32,
                                                                  time_major=True)

            rnn_final_state = tf.concat(1, [final_states[0].h, final_states[1].h])
            rnn_final_state = tf.nn.dropout(x=rnn_final_state, keep_prob=self.keep_prob_)
            # assuming binary classification
            # prediction_logit = tf.contrib.layers.fully_connected(inputs=rnn_final_state, num_outputs=1,
            #                                                      trainable=True, activation_fn=None,
            #                                                      biases_initializer=tf.zeros_initializer)

            rnn_logit = tf.contrib.layers.fully_connected(inputs=rnn_final_state, num_outputs=1,
                                                          trainable=True, activation_fn=None,
                                                          biases_initializer=tf.zeros_initializer)

            normalized_track_len = (tf.reshape(tf.cast(self.track_len_, tf.float32),
                                               [1, 1]) - track_len_mean) / track_len_std
            normalized_points_within_50m_count = (tf.reshape(tf.cast(self.points_within_50m_count_, tf.float32),
                                                             [1,
                                                              1]) - points_within_50m_mean_count) / points_within_50m_std_count

            final_initial_state = tf.concat(1, [rnn_logit, normalized_track_len, normalized_points_within_50m_count])
            prediction_logit = tf.contrib.layers.fully_connected(inputs=final_initial_state, num_outputs=1,
                                                                 trainable=True, activation_fn=None,
                                                                 biases_initializer=tf.zeros_initializer)
            self.prediction_ = tf.nn.sigmoid(prediction_logit)
            self.loss_ = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_logit,
                                                                 targets=tf.cast(self.target_class_, tf.float32))
            self.optimizer_ = self._get_optimizer(self.loss_, learning_rate)
            self.saver_ = tf.train.Saver()

    @staticmethod
    def _count_of_points_within_50m(x):
        logging.debug('x.shape: {}'.format(x.shape))
        return len(np.argwhere(x[:, 0] <= 50).ravel())

    def predict_proba(self, x):
        feed_dict = {self.track_len_: np.array([len(x)]),
                     self.input_track_: np.reshape(x, [-1, 1, self.nn_params.n_dim]),
                     self.points_within_50m_count_: np.array([self._count_of_points_within_50m(x)])}
        return self.session_.run([self.prediction_], feed_dict=feed_dict)[0].ravel()

    def predict(self, x):
        return np.int(np.rint(self.predict_proba(x)))


def interaction_recs_2_matrix(interaction_recs):
    df = pd.DataFrame(interaction_recs)
    df.index = df['timestamp'].map(pd.datetime.utcfromtimestamp)
    df.sort_index(inplace=True)
    return df[['dist', 'speed']].resample('10S').mean().interpolate().as_matrix()


class NNEventTypeDeterminer(EventTypeDeterminerBase):
    def __init__(self, params):
        super(NNEventTypeDeterminer, self).__init__(params)
        self.params.max_definitely_short_interaction_duration = 15
        self.params.probably_continuous_interaction_rings_conf = [
            [
                (0, 30),
                (30, 60),
                (60, 70)
            ],
            [
                (0, 15),
                (15, 45),
                (45, 70)
            ],
            [
                (0, 5),
                (5, 35),
                (35, 65),
                (65, 70)
            ]
        ]
        self.params.probably_continuous_interaction_min_ring_stay_time = 60
        self._nn_estimator = BiDirBinaryDynamicLSTMPredictor(NNParams, 'stat.json', './rnn-model')

    def _is_definitely_short_interaction(self, interaction):
        return interaction.duration < self.params.max_definitely_short_interaction_duration

    def _is_probably_continuous_interaction(self, interaction):
        time_index = [r['timestamp'] for r in interaction.records]
        dist_series = [r['dist'] for r in interaction.records]

        interp_time_index = np.arange(time_index[0], time_index[-1], step=5)
        interp_values = np.interp(interp_time_index, time_index, dist_series)
        interp_series = [{'timestamp': ts, 'dist': dist} for ts, dist in zip(interp_time_index, interp_values)]

        def calc_hist(conf):
            def dist_to_r(dist, radiuses):
                """maps dist to the nearest above radius"""
                for r_min, r_max in radiuses:
                    if r_min <= dist < r_max:
                        return r_max

            def track_duration(track):
                return track[-1]['timestamp'] - track[0]['timestamp']

            hist = defaultdict(int)

            for r, points in groupby(interp_series, key=lambda x: dist_to_r(x['dist'], conf)):
                if r is not None:
                    points = list(points)
                    hist[r] += track_duration(points)

            return hist

        for ring in self.params.probably_continuous_interaction_rings_conf:
            if any(v > self.params.probably_continuous_interaction_min_ring_stay_time
                   for v in calc_hist(ring).itervalues()):
                return True

        return False

    def _is_continuous_interaction_by_nn(self, interaction):
        x = interaction_recs_2_matrix(interaction.records)
        assert x.shape[1] == 2, 'invalid x shape, {}'.format(x.shape)
        return bool(self._nn_estimator.predict(x))

    def _determine_event_type(self, interaction):
        if self._is_definitely_short_interaction(interaction):
            min_dist_rec = min([r for r in interaction.records], key=operator.itemgetter('dist'))

            if min_dist_rec['dist'] <= self.params.short_interaction_r:
                return Event(
                    interaction.point_id,
                    EventType.SHORT,
                    min_dist_rec,
                    min_dist_rec
                )

        if self._is_probably_continuous_interaction(interaction) and self._is_continuous_interaction_by_nn(interaction):
            if interaction.duration < self.params.max_continuous_interaction_time:
                event_type = EventType.CONTINUOUS
            else:
                event_type = EventType.SUPER

            return Event(
                interaction.point_id,
                event_type,
                interaction.records[0],
                interaction.records[-1]
            )

        short_interaction_before_loss = self._get_short_interaction_before_loss(interaction)
        if short_interaction_before_loss:
            return short_interaction_before_loss

        min_dist_rec = min([r for r in interaction.records], key=operator.itemgetter('dist'))

        if min_dist_rec['dist'] <= self.params.short_interaction_r:
            return Event(
                interaction.point_id,
                EventType.SHORT,
                min_dist_rec,
                min_dist_rec
            )
