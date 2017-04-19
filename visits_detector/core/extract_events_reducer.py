from __future__ import division

from collections import defaultdict

from visits_detector.core.components.event_type_determiner import HeuristicsEventTypeDeterminer, NNEventTypeDeterminer
from visits_detector.core.components.interaction_splitter.interaction_splitter import InteractionSplitter


class ExtractEventsReducer(object):
    def __init__(self, params):
        self.params = params
        self._interaction_splitter = InteractionSplitter(self.params)
        if self.params.use_nn_estimator:
            self._event_type_determiner = NNEventTypeDeterminer(self.params)
        else:
            self._event_type_determiner = HeuristicsEventTypeDeterminer(self.params)

    def extract_events(self, recs):
        point_id_events = defaultdict(list)

        for interaction in self._interaction_splitter(recs):
            event = self._event_type_determiner(interaction)

            if not event:
                continue

            if point_id_events[interaction.point_id]:
                last_event_timestamp = point_id_events[interaction.point_id][-1].end_ts
                if event.start_ts - last_event_timestamp > self.params.event_timeout:
                    yield self._event_type_determiner.choose_best_event(point_id_events[interaction.point_id]) \
                        .to_event_rec()
                    point_id_events[interaction.point_id] = []

            point_id_events[interaction.point_id].append(event)

        for point_id, events in point_id_events.iteritems():
            yield self._event_type_determiner.choose_best_event(events).to_event_rec()

    def __call__(self, rec_groups):
        if self.params.use_nn_estimator:
            self._event_type_determiner = NNEventTypeDeterminer(self.params)
        else:
            self._event_type_determiner = HeuristicsEventTypeDeterminer(self.params)

        for key, recs in rec_groups:
            id_ = key[self.params.id_column]

            for event in self.extract_events(recs):
                event[self.params.id_column] = id_
                yield event
