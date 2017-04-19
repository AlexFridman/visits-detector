import argparse
import datetime
import json
import sys

import pyspark

from visits_detector.core import FilterAndMapToIndexMapper, ExtractEventsReducer
from visits_detector.core.components.params import EventExtractionStageParams
from visits_detector.core.helpers.geo_index import build_geo_index_from_point_index
from visits_detector.core.helpers.index import get_bbox_by_index, read_index


def lazy_load_gps_log(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-path', required=True)
    parser.add_argument('--gps-log-path', required=True)

    parser.add_argument('--use-nn', action='store_true')
    parser.add_argument('--model-path')

    return parser.parse_args()


def spark_mapper_wrapper(mapper):
    def inner(rec):
        return list(mapper([rec]))

    return inner


def spark_reducer_wrapper(reducer_inst_fn, reduce_by=None, sort_by=None):
    def inner(key_recs):
        key, recs = key_recs
        if sort_by:
            recs = sorted(recs, key=lambda x: tuple(x.get(f) for f in sort_by))

        key = dict(zip(reduce_by, key))
        return list(reducer_inst_fn()([[key, recs]]))

    return inner


def main():
    args = parse_args()

    index = read_index(args.index_path)

    mapper = FilterAndMapToIndexMapper(
        start_dt=datetime.datetime.min,
        end_dt=datetime.datetime.max,
        bbox=get_bbox_by_index(index, 0.1),
        geo_index=build_geo_index_from_point_index(index, precision=6),
        cut_off_r=200,
        id_column='uuid',
        timestamp_column='timestamp',
        speed_column='speed',
        lat_column='lat',
        lon_column='lon'
    )
    mapper = spark_mapper_wrapper(mapper)

    reducer_params = EventExtractionStageParams(
        use_nn_estimator=args.use_nn,
        model_path=args.model_path
    )
    reducer_inst_fn = lambda: ExtractEventsReducer(reducer_params)
    reducer = spark_reducer_wrapper(reducer_inst_fn, ['uuid'], ['timestamp'])

    sc = pyspark.SparkContext('local[*]')

    events = sc.textFile(args.gps_log_path) \
        .map(json.loads) \
        .flatMap(mapper) \
        .map(lambda x: (x['uuid'], [x])) \
        .reduceByKey(lambda _1, _2: _1 + _2) \
        .flatMap(reducer) \
        .collect()

    for event in events:
        sys.stdout.write(json.dumps(event) + '\n')


if __name__ == '__main__':
    main()
