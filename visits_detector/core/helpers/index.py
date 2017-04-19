import json

import cyson


def read_index(path):
    try:
        with open(path) as fp:
            return {r['id']: r for r in json.load(fp)}
    except:
        pass

    with open(path) as fp:
        return {r['id']: r for r in map(json.loads, fp)}


def build_coord_index(index):
    return {id_: (point['latitude'], point['longitude']) for id_, point in index.iteritems()}


def get_bbox_by_index(index, delta):
    min_lat = min_lon = 181
    max_lat = max_lon = -1

    for _, point in index.iteritems():
        lat, lon = point.get('latitude', point.get('lat')), point.get('longitude', point.get('lon'))

        if lat < min_lat:
            min_lat = lat

        if lat > max_lat:
            max_lat = lat

        if lon < min_lon:
            min_lon = lon

        if lon > max_lon:
            max_lon = lon

    return [(min_lat - delta, min_lon - delta), (max_lat + delta, max_lon + delta)]
