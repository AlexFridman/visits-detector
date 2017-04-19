import operator

from visits_detector.core.components.data_structures import EventType, Event, SubInteraction
from visits_detector.core.components.event_type_determiner.event_type_determiner_base import EventTypeDeterminerBase


class HeuristicsEventTypeDeterminer(EventTypeDeterminerBase):
    @staticmethod
    def _choose_longest_sub_interaction(sub_interactions):
        return max(sub_interactions, key=lambda x: x.duration)

    def _split_interaction_into_sub_interaction(self, interaction, continuous_interaction_r):
        sub_interaction = None
        # add dummy record to avoid losses of continuous interactions with last record in ci radius
        for rec in interaction.records:
            # the first iteration
            if sub_interaction is None:
                if rec['dist'] <= continuous_interaction_r:
                    sub_interaction = SubInteraction(rec, rec)
            elif rec['dist'] <= continuous_interaction_r:
                # got another point from interaction series
                sub_interaction.end_rec = rec
            # not the first iteration and got a point out of the allowed radius
            elif sub_interaction:
                yield sub_interaction
                sub_interaction = None
            else:
                raise RuntimeError('bug')

        if sub_interaction:
            yield sub_interaction

    def _determine_event_type(self, interaction):
        continuous_interaction_r = self.params.continuous_interaction_r
        interaction_wo_outliers = interaction.drop_small_outliers_intervals(continuous_interaction_r,
                                                                            self.params.merge_sub_interactions_time_delta)

        sub_interactions = self._split_interaction_into_sub_interaction(interaction_wo_outliers,
                                                                        continuous_interaction_r)
        at_least_continuous_sub_interactions = [x for x in sub_interactions
                                                if x.duration >= self.params.min_continuous_interaction_time]

        if at_least_continuous_sub_interactions:
            longest_sub_interaction = self._choose_longest_sub_interaction(at_least_continuous_sub_interactions)

            if longest_sub_interaction.duration < self.params.max_continuous_interaction_time:
                event_type = EventType.CONTINUOUS
            else:
                event_type = EventType.SUPER

            return Event(
                interaction.point_id,
                event_type,
                longest_sub_interaction.start_rec,
                longest_sub_interaction.end_rec
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
