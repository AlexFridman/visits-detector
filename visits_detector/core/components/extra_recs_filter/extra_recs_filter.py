from itertools import groupby


def extra_recs_filter(recs):
    """
    Assume recs are sorted by timestamp
    """
    for is_nearest_points_not_empty, group_recs in groupby(recs, key=lambda x: bool(x['nearest_points'])):
        if is_nearest_points_not_empty:
            for rec in group_recs:
                yield rec
        else:
            first = last = next(group_recs)
            yield first

            for last in group_recs:
                pass

            if last['timestamp'] > first['timestamp']:
                yield last
