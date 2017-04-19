import datetime
from collections import namedtuple
from itertools import groupby

from visits_detector.core.common import DATE_FORMAT


class Interaction(namedtuple('Interaction', ['point_id', 'records', 'timestamp', 'time_before_next_interaction'])):
    @property
    def duration(self):
        return self.records[-1]['timestamp'] - self.records[0]['timestamp']

    def drop_small_outliers_intervals(self, radius, max_outlier_interval_len):
        new_recs = []

        for is_outlier, recs in groupby(self.records, lambda x: x['dist'] > radius):
            recs = list(recs)

            if is_outlier and recs[-1]['timestamp'] - recs[0]['timestamp'] <= max_outlier_interval_len:
                continue

            new_recs.extend(recs)

        return Interaction(
            point_id=self.point_id,
            records=new_recs or self.records,
            timestamp=self.timestamp,
            time_before_next_interaction=self.time_before_next_interaction
        )


class SubInteraction(object):
    __slots__ = ['start_rec', 'end_rec']

    def __init__(self, start_rec, end_rec):
        self.start_rec = start_rec
        self.end_rec = end_rec

    @property
    def start_ts(self):
        return self.start_rec['timestamp']

    @property
    def end_ts(self):
        return self.end_rec['timestamp']

    @property
    def duration(self):
        return self.end_ts - self.start_ts


class EventType(object):
    CONTINUOUS = 'continuous_interaction'
    SUPER = 'super_interaction'
    SHORT = 'short_interaction'
    SHORT_BEFORE_LOSS = 'short_interaction_before_loss'


class Event(SubInteraction):
    def __init__(self, point_id, event_type, start_rec, end_rec):
        super(Event, self).__init__(start_rec, end_rec)
        self.point_id = point_id
        self.event_type = event_type

    def to_event_rec(self):
        if self.event_type in {EventType.SUPER, EventType.CONTINUOUS}:
            return {
                'event': self.event_type,
                'lat_0': self.start_rec['lat'],
                'lon_0': self.start_rec['lon'],
                'lat_1': self.end_rec['lat'],
                'lon_1': self.end_rec['lon'],
                'lat': self.start_rec['lat'],
                'lon': self.start_rec['lon'],
                'timestamp': (self.start_ts + self.end_ts) / 2,
                'timestamp_0': self.start_ts,
                'timestamp_1': self.end_ts,
                'dist_0': self.start_rec['dist'],
                'dist_1': self.end_rec['dist'],
                'point_id': self.point_id,
                'date': datetime.datetime.utcfromtimestamp(self.start_ts).strftime(DATE_FORMAT),
                'spent_time': self.duration

            }
        return {
            'event': self.event_type,
            'lat': self.start_rec['lat'],
            'lon': self.start_rec['lon'],
            'timestamp': self.start_ts,
            'dist': self.start_rec['dist'],
            'point_id': self.point_id,
            'date': datetime.datetime.utcfromtimestamp(self.start_ts).strftime(DATE_FORMAT)
        }
