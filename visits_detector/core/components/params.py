class EventExtractionStageParams(object):
    def __init__(self,
                 id_column='uuid',
                 cut_off_r=200,
                 short_interaction_r=100,
                 continuous_interaction_r=30,
                 si_before_loss_r=100,
                 merge_sub_interactions_time_delta=20,
                 min_continuous_interaction_time=90,
                 max_continuous_interaction_time=900,
                 event_timeout=10000,
                 min_time_in_si_r_before_loss=15,
                 min_track_timeout_after_loss=90,
                 session_timeout=10000,
                 max_interaction_break=20,
                 default_dist=200,
                 use_nn_estimator=False):
        """
        :param session_timeout: time in seconds between two sessions
        :param max_interaction_break: time in seconds between two nearest interactions
        :param default_dist: default dist
        """
        self.id_column = id_column
        self.cut_off_r = cut_off_r
        self.short_interaction_r = short_interaction_r
        self.continuous_interaction_r = continuous_interaction_r
        self.si_before_loss_r = si_before_loss_r
        self.merge_sub_interactions_time_delta = merge_sub_interactions_time_delta
        self.min_continuous_interaction_time = min_continuous_interaction_time
        self.max_continuous_interaction_time = max_continuous_interaction_time
        self.event_timeout = event_timeout
        self.min_time_in_si_r_before_loss = min_time_in_si_r_before_loss
        self.min_track_timeout_after_loss = min_track_timeout_after_loss
        self.session_timeout = session_timeout
        self.max_interaction_break = max_interaction_break
        self.default_dist = default_dist
        self.use_nn_estimator = use_nn_estimator
