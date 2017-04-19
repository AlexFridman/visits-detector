from __future__ import division

from geoindex import GeoGridIndex, GeoPoint


def build_geo_index_from_coord_index(coord_index, precision=5):
    geo_index = GeoGridIndex(precision=precision)

    for (lat, lng) in coord_index:
        geo_index.add_point(GeoPoint(lat, lng))

    return geo_index


def build_geo_index_from_point_index(index, precision=5):
    geo_index = GeoGridIndex(precision=precision)

    for id_, point_info in index.iteritems():
        lat, lon = point_info.get('latitude', point_info.get('lat')), point_info.get('longitude', point_info.get('lon'))
        geo_index.add_point(GeoPoint(lat, lon, point_info['id']))

    return geo_index


def get_nearest_point(geo_index, lat, lng, r):
    nearest_points_dists = list(geo_index.get_nearest_points(GeoPoint(lat, lng), r / 1000))
    if nearest_points_dists:
        nearest_point, dist = min(nearest_points_dists, key=lambda x: x[1])
        dist *= 1000
        if dist <= r:
            return nearest_point.ref, dist
    return None, None


def get_nearest_points(geo_index, lat, lng, r):
    for point, dist in geo_index.get_nearest_points(GeoPoint(lat, lng), r / 1000):
        dist *= 1000
        if dist <= r:
            yield point.ref, dist
