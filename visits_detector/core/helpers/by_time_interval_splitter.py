def split_by_time_interval(recs, timeout, timestamp_field='timestamp'):
    curr_trip = None
    prev_ts = -10 ** 10

    for rec in recs:
        ts = rec[timestamp_field]
        if ts - prev_ts > timeout:
            if curr_trip:
                yield curr_trip
            curr_trip = []

        curr_trip.append(rec)
        if ts < prev_ts:
            raise RuntimeError('Records should be sorted by time')

        prev_ts = ts

    if curr_trip:
        yield curr_trip
