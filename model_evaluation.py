# -*- coding: utf-8 -*-
from training_phase import *


def evaluate_training_and_test_set():
    """This function should compute the accuracy score of your model
       on the training data
       Returns
       -------
       accuracy_score: type-tuple of floats
       """
    w_ = train_classifier()
    p = Perceptron()
    p.w_ = w_

    accuracy = lambda X, y: sum(y == p.predict(x)
                                for x, y in zip(X, y)) / float(len(y))

    return (accuracy(X, y) for X, y in
            [read_featurize_set(), read_featurize_set('test')])
