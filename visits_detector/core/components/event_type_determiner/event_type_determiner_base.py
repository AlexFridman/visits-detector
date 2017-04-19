import sys
from abc import ABCMeta, abstractmethod
from itertools import chain

from visits_detector.core.components.data_structures import Event, EventType
from visits_detector.core.helpers.by_time_interval_splitter import split_by_time_interval


class EventTypeDeterminerBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, params):
        self.params = params

    def _get_short_interaction_before_loss(self, interaction):
        si_before_loss_r = self.params.si_before_loss_r

        dummy_record = {
            'timestamp': interaction.records[-1]['timestamp'] + interaction.time_before_next_interaction + 1,
            'dist': sys.maxint
        }

        recs = chain(interaction.records, [dummy_record])
        for sub_interaction in split_by_time_interval(recs, timeout=self.params.min_track_timeout_after_loss):
            last_si_r_entrance = sub_interaction[-1]['timestamp']
            start_rec = sub_interaction[-1]

            for r in reversed(sub_interaction):
                if r['dist'] > si_before_loss_r:
                    break
                last_si_r_entrance = r['timestamp']
                start_rec = r

            time_in_si_r_before_loss = sub_interaction[-1]['timestamp'] - last_si_r_entrance
            if time_in_si_r_before_loss >= self.params.min_time_in_si_r_before_loss:
                return Event(
                    point_id=interaction.point_id,
                    event_type=EventType.SHORT_BEFORE_LOSS,
                    start_rec=start_rec,
                    end_rec=sub_interaction[-1]
                )

    @abstractmethod
    def _determine_event_type(self, interaction):
        pass

    def __call__(self, interaction):
        return self._determine_event_type(interaction)

    @staticmethod
    def choose_best_event(events):
        priority = {
            EventType.SUPER: 0,
            EventType.CONTINUOUS: 1,
            EventType.SHORT_BEFORE_LOSS: 2,
            EventType.SHORT: 3
        }

        return sorted(events, key=lambda x: priority[x.event_type])[0]
