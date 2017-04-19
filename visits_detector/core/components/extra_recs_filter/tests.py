import unittest

from visits_detector.core.components.extra_recs_filter import extra_recs_filter


class ExtraRecsFilterTestCase(unittest.TestCase):
    @staticmethod
    def format_rec(rec):
        ts, nearest_points = rec
        return {
            'timestamp': ts,
            'nearest_points': nearest_points
        }

    def test_1(self):
        test_data = [
            [0, {}],
            [10, {}],
            [20, {}],
            [30, {0: 200}],
            [40, {0: 190}],
            [50, {0: 150}],
            [60, {0: 40}],
            [70, {}],
            [80, {}],
            [90, {}],
            [100, {0: 40}],

        ]

        recs = list(extra_recs_filter(map(self.format_rec, test_data)))
        self.assertEqual(len(recs), 9)
