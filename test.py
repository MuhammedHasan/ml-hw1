import unittest

from feature_extraction import *
from training_phase import *
from model_evaluation import *

# If you want to see visualziation use this lines
# visualise_features()
# visualize_decision_boundary()


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


class TestModelEvaluation(unittest.TestCase):

    def test_feature1(self):
        (train_set, test_set) = evaluate_training_and_test_set()
        self.assertTrue(0.1 > train_set - 0.88)
        self.assertTrue(0.1 > test_set - 0.81)


class TestTrainingPhase(unittest.TestCase):

    def test_train_classifier(self):
        w_ = train_classifier()
        self.assertTrue(1 > 24.32 - w_[0])
        self.assertTrue(5 > -108.73 - w_[1])
        self.assertTrue(1 > 0.94 - w_[2])

if __name__ == '__main__':
    unittest.main()
