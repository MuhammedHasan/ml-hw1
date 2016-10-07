import unittest
from feature_extraction import *
from training_phase import *


class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        self.matrix = [
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ]

    def test_feature1(self):
        self.assertEqual(feature1(self.matrix), 0.375)

    def test_feature2(self):
        self.assertEqual(feature2(self.matrix), 15)

if __name__ == '__main__':
    unittest.main()
