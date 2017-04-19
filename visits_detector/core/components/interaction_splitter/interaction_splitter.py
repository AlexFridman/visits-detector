import sys

from visits_detector.core.components.data_structures import Interaction
from visits_detector.core.components.extra_recs_filter import extra_recs_filter


class InteractionEnd(Exception):
    def __init__(self, interaction):
        super(InteractionEnd, self).__init__('interaction end')
        self.interaction = interaction


class InteractionBorder(InteractionEnd):
    pass


class RecordStreamEnd(Exception):
    pass


def coroutine(f):
    def wrap(*args, **kwargs):
        gen = f(*args, **kwargs)
        gen.send(None)
        return gen

    return wrap


def create_interaction_splitter_coroutine_type(params):
    @coroutine
    def interaction_coroutine(point_id):
        interaction = []
        is_interaction_break = False
        recs_wo_point_id = []
        prev_rec_timestamp = None

        def is_session_break():
            assert interaction
            return rec['timestamp'] - prev_rec_timestamp > params.session_timeout

        def is_break_too_long():
            assert interaction
            last_interaction_ts = interaction[-1]['timestamp']
            return is_interaction_break and rec['timestamp'] - last_interaction_ts > params.max_interaction_break

        def prettify_interaction(stream_end=False):
            pretty_recs = []

            for r in interaction:
                pretty_recs.append({
                    'timestamp': r['timestamp'],
                    'dist': r['nearest_points'].get(point_id, params.default_dist),
                    'speed': r['speed'],
                    'lat': r['lat'],
                    'lon': r['lon']
                })

            _1st_rec_after_interaction_ts = min(rec['timestamp'],
                                                min([r['timestamp'] for r in recs_wo_point_id] or [sys.maxint]))
            time_before_next_interaction = _1st_rec_after_interaction_ts - pretty_recs[-1]['timestamp']

            if stream_end:
                time_before_next_interaction = time_before_next_interaction or sys.maxint

            return Interaction(
                point_id=point_id,
                records=pretty_recs,
                timestamp=pretty_recs[0]['timestamp'],
                time_before_next_interaction=time_before_next_interaction
            )

        while True:
            try:
                rec = (yield)

                if rec is None:
                    continue

                if prev_rec_timestamp is None:
                    prev_rec_timestamp = rec['timestamp']

                if rec['timestamp'] < prev_rec_timestamp:
                    raise RuntimeError('Records should be sorted by time')

                if point_id in rec['nearest_points']:
                    if interaction and (is_break_too_long() or is_session_break()):
                        raise InteractionBorder(prettify_interaction())

                    if recs_wo_point_id:
                        interaction.extend(recs_wo_point_id)
                        recs_wo_point_id = []

                    interaction.append(rec)

                    is_interaction_break = False
                elif is_break_too_long():
                    raise InteractionEnd(prettify_interaction())
                else:
                    is_interaction_break = True
                    recs_wo_point_id.append(rec)

                prev_rec_timestamp = rec['timestamp']
            except RecordStreamEnd:
                raise InteractionEnd(prettify_interaction(stream_end=True))

    return interaction_coroutine


class InteractionSplitter(object):
    def __init__(self, params):
        self._interaction_splitter_coroutine_type = create_interaction_splitter_coroutine_type(params)

    def __call__(self, recs):
        point_coroutine_map = {}

        for rec in extra_recs_filter(recs):
            for point_id, dist in rec['nearest_points'].iteritems():
                if point_id not in point_coroutine_map:
                    point_coroutine_map[point_id] = self._interaction_splitter_coroutine_type(point_id)

            # sorted is used just for asserts simplification
            for point_id, coroutine in sorted(point_coroutine_map.iteritems()):
                try:
                    coroutine.send(rec)
                except InteractionBorder as e:
                    yield e.interaction
                    point_coroutine_map[point_id] = self._interaction_splitter_coroutine_type(point_id)
                    point_coroutine_map[point_id].send(rec)
                except InteractionEnd as e:
                    del point_coroutine_map[point_id]
                    yield e.interaction

        for point_id, coroutine in point_coroutine_map.iteritems():
            try:
                coroutine.throw(RecordStreamEnd)
            except InteractionEnd as e:
                yield e.interaction
