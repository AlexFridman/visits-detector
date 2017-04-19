import sys
import unittest

from visits_detector.core.components.interaction_splitter.interaction_splitter import InteractionSplitter
from visits_detector.core.components.params import EventExtractionStageParams


class InteractionSplitterTestCase(unittest.TestCase):
    params = EventExtractionStageParams(session_timeout=20)

    @staticmethod
    def format_rec(rec):
        ts, nearest_points = rec
        return {
            'timestamp': ts,
            'nearest_points': nearest_points,
            'speed': None,
            'lat': 0,
            'lon': 0
        }

    def test_1(self):
        test_data = [
            [0, {}],
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [40, {0: 40}],
            [50, {0: 40}],
            [60, {0: 40}],
            [70, {0: 120}],
            [80, {0: 170}],
            [90, {0: 180}],
            [100, {0: 200}],
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 1)
        self.assertEqual(len(interactions[0].records), 10)

    def test_2(self):
        test_data = [
            [0, {}],
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [40, {0: 40}],
            ###############
            [70, {0: 40}],
            [80, {0: 40}],
            [90, {0: 120}],
            [100, {0: 170}],
            [110, {0: 180}],
            [120, {0: 200}],
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 2)
        self.assertEqual(len(interactions[0].records), 4)
        self.assertEqual(len(interactions[1].records), 6)

    def test_3(self):
        test_data = [
            [0, {}],
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [40, {0: 40}],
            ###############
            [70, {0: 40}],
            ###############
            [100, {0: 40}],
            [110, {0: 120}],
            [120, {0: 170}],
            [130, {0: 180}],
            [140, {0: 200}],
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 3)
        self.assertEqual(len(interactions[0].records), 4)
        self.assertEqual(len(interactions[1].records), 1)
        self.assertEqual(len(interactions[2].records), 5)

    def test_4(self):
        test_data = [
            [0, {}],
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [40, {0: 40, 1: 100}],

            [55, {1: 80}],

            [70, {0: 40, 1: 40}],

            [85, {1: 20}],

            [100, {0: 40, 1: 40}],
            [110, {0: 120, 1: 100}],
            [120, {0: 170}],
            [130, {0: 180}],
            [140, {0: 200}],
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 4)
        self.assertEqual(len(interactions[0].records), 4)
        self.assertEqual(len(interactions[1].records), 1)
        self.assertEqual(len(interactions[2].records), 6)
        self.assertEqual(len(interactions[3].records), 5)

    def test_5(self):
        test_data = [
            [0, {}],
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [40, {0: 40, 1: 100}],

            [55, {1: 80}],

            [70, {0: 40, 1: 40}],

            [85, {1: 20}],

            [100, {0: 40, 1: 40}],
            [110, {0: 120, 1: 100}],
            [120, {0: 170}],
            [130, {}],
            [150, {}]
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 4)
        self.assertEqual(len(interactions[0].records), 4)
        self.assertEqual(len(interactions[1].records), 1)
        self.assertEqual(len(interactions[2].records), 3)
        self.assertEqual(len(interactions[3].records), 6)

    def test_6(self):
        test_data = [
            [0, {}],
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [40, {0: 40, 1: 100}],

            [50, {}],

            [55, {1: 80}],

            [70, {0: 40, 1: 40}],

            [85, {1: 20}],

            [100, {0: 40, 1: 40}],
            [110, {0: 120, 1: 100}],
            [120, {0: 170}],
            [130, {}],
            [150, {}]
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 4)
        self.assertEqual(len(interactions[0].records), 4)
        self.assertEqual(len(interactions[1].records), 1)
        self.assertEqual(len(interactions[2].records), 3)
        self.assertEqual(len(interactions[3].records), 7)

        self.assertEqual(interactions[0].time_before_next_interaction, 10)
        self.assertEqual(interactions[1].time_before_next_interaction, 15)
        self.assertEqual(interactions[2].time_before_next_interaction, 10)
        self.assertEqual(interactions[3].time_before_next_interaction, 10)

    def test_7(self):
        test_data = [
            [0, {}],
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [40, {0: 40}],
            [50, {0: 40}],
            [60 + self.params.session_timeout, {0: 40}],
            [70 + self.params.session_timeout, {0: 120}],
            [80 + self.params.session_timeout, {0: 170}],
            [90 + self.params.session_timeout, {0: 180}],
            [100 + self.params.session_timeout, {0: 200}],
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 2)
        self.assertEqual(len(interactions[0].records), 5)
        self.assertEqual(len(interactions[0].records), 5)

    def test_8(self):
        test_data = [
            [0, {}],
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [40, {0: 40}],
            [50, {0: 40}],
            [self.params.session_timeout / 2, {}],
            [60 + self.params.session_timeout, {0: 40}],
            [70 + self.params.session_timeout, {0: 120}],
            [80 + self.params.session_timeout, {0: 170}],
            [90 + self.params.session_timeout, {0: 180}],
            [100 + self.params.session_timeout, {0: 200}],
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 2)
        self.assertEqual(len(interactions[0].records), 5)
        self.assertEqual(len(interactions[0].records), 5)

    def test_9(self):
        test_data = [
            [0, {}],
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [35, {}],
            [40, {0: 40}],
            [50, {0: 40}],
            [60 + self.params.session_timeout, {0: 40}],
            [70 + self.params.session_timeout, {0: 120}],
            [80 + self.params.session_timeout, {0: 170}],
            [90 + self.params.session_timeout, {0: 180}],
            [100 + self.params.session_timeout, {0: 200}],
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 2)
        self.assertEqual(len(interactions[0].records), 6)
        self.assertEqual(len(interactions[1].records), 5)

        self.assertEqual(interactions[0].time_before_next_interaction, 60 + self.params.session_timeout - 50)
        self.assertEqual(interactions[1].time_before_next_interaction, sys.maxint)

    def test_10(self):
        test_data = [
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 1)
        self.assertEqual(len(interactions[0].records), 3)
        self.assertEqual(interactions[0].time_before_next_interaction, sys.maxint)

    def test_11(self):
        test_data = [
            [10, {0: 200}],
            [20, {0: 190}],
            [30, {0: 150}],
            [50, {}],
        ]

        splitter = InteractionSplitter(self.params)
        interactions = list(splitter(map(self.format_rec, test_data)))
        self.assertEqual(len(interactions), 1)
        self.assertEqual(len(interactions[0].records), 3)
        self.assertEqual(interactions[0].time_before_next_interaction, 20)
