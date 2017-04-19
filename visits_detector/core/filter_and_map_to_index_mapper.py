import datetime

from visits_detector.core.common import DATE_FORMAT
from visits_detector.core.helpers.geo_index import get_nearest_points


class FilterAndMapToIndexMapper(object):
    """
    Filter and map records to all points in short_interaction_r
    input table format: (lat_column, lon_column, id_column, timestamp_column, speed_column)
    """

    def __init__(self,
                 start_dt,
                 end_dt,
                 bbox,
                 geo_index,
                 cut_off_r,
                 id_column,
                 timestamp_column,
                 speed_column,
                 lat_column,
                 lon_column):
        self._start_dt = start_dt
        self._end_dt = end_dt
        self._bbox = bbox
        self._geo_index = geo_index
        self._id_column = id_column
        self._cut_off_r = cut_off_r
        self._timestamp_column = timestamp_column
        self._speed_column = speed_column
        self._lat_column = lat_column
        self._lon_column = lon_column

    def _is_in_bbox(self, lat, lon):
        (min_lat, min_lon), (max_lat, max_lon) = self._bbox

        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

    def _filter_recs(self, recs):
        for rec in recs:
            ts = int(rec[self._timestamp_column])
            dt = datetime.datetime.utcfromtimestamp(ts)

            lat, lon = float(rec[self._lat_column]), float(rec[self._lon_column])

            if self._start_dt <= dt <= self._end_dt and self._is_in_bbox(lat, lon):
                yield {
                    self._id_column: rec[self._id_column],
                    'timestamp': ts,
                    'date': dt.strftime(DATE_FORMAT),
                    'lat': lat,
                    'lon': lon,
                    'speed': float(rec[self._speed_column])
                }

    def __call__(self, recs):
        for rec in self._filter_recs(recs):
            rec['nearest_points'] = dict(list(get_nearest_points(self._geo_index, rec['lat'], rec['lon'],
                                                                 self._cut_off_r)))
            yield rec
