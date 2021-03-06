import argparse
import datetime
import json
import sys

from visits_detector.core import FilterAndMapToIndexMapper, ExtractEventsReducer
from visits_detector.core.components.params import EventExtractionStageParams
from visits_detector.core.helpers.geo_index import build_geo_index_from_point_index
from visits_detector.core.helpers.index import get_bbox_by_index, read_index
from visits_detector.demo.mr_emulator import MapReduceEmulator


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


def main():
    args = parse_args()

    index = read_index(args.index_path)
    gps_log = lazy_load_gps_log(args.gps_log_path)

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

    reducer_params = EventExtractionStageParams(
        use_nn_estimator=args.use_nn,
        model_path=args.model_path
    )
    reducer = ExtractEventsReducer(reducer_params)

    mr_emulator = MapReduceEmulator(mapper, reducer, sort_by=['uuid', 'timestamp'], reduce_by=['uuid'])

    for event in mr_emulator(gps_log):
        sys.stdout.write(json.dumps(event) + '\n')


if __name__ == '__main__':
    main()
