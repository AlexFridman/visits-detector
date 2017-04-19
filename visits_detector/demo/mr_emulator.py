from itertools import groupby


class MapReduceEmulator(object):
    @staticmethod
    def _dummy_mapper(recs):
        for rec in recs:
            yield rec

    @staticmethod
    def _dummy_reducer(groups):
        for key, recs in groups:
            for rec in recs:
                yield rec

    @staticmethod
    def _key_func(key_fields, rec):
        return tuple(rec.get(f) for f in key_fields)

    def _groupby(self, key_fields, stream):
        for key, recs in groupby(stream, key=lambda x: self._key_func(key_fields, x)):
            yield dict(zip(key_fields, key)), recs

    def __init__(self, mapper=None, reducer=None, sort_by=None, reduce_by=None):
        self.mapper = mapper or self._dummy_mapper
        self.reducer = reducer or self._dummy_reducer
        self.sort_by = sort_by or []
        self.reduce_by = reduce_by or []

    def __call__(self, stream):
        mapped = self.mapper(stream)
        sorted_ = sorted(mapped, key=lambda x: self._key_func(self.sort_by, x))
        groupped = self._groupby(self.reduce_by, sorted_)
        for rec in self.reducer(groupped):
            yield rec
