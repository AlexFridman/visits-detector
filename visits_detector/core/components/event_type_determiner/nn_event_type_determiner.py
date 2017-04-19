# !/usr/bin/env python
# -*- coding: utf-8 -*-
import operator
from collections import defaultdict
from itertools import groupby

import numpy as np
import pandas as pd

from visits_detector.core.components.data_structures import EventType, Event
from visits_detector.core.components.event_type_determiner.event_type_determiner_base import EventTypeDeterminerBase
from visits_detector.core.nn_model.model import NNModel


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
        self._nn_estimator = NNModel(self.params.model_path)

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
