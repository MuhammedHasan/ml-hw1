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

    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))

    train_positives = [(1, x) for _, x in train_positives.items()]
    train_negatives = [(-1, x) for _, x in train_negatives.items()]

    train = train_positives + train_negatives

    Xy = np.array([[feature1(v), feature2(v), k] for k, v in train])
    X = Xy[:, :2]
    y = Xy[:, -1]

    accuracy_on_training_set = sum(y == p.predict(x) for x, y in zip(X, y))
    accuracy_on_training_set /= float(len(train))

    train_positives = pickle.load(open('test_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('test_set_negatives.p', 'rb'))

    train_positives = [(1, x) for _, x in train_positives.items()]
    train_negatives = [(-1, x) for _, x in train_negatives.items()]

    train = train_positives + train_negatives

    Xy = np.array([[feature1(v), feature2(v), k] for k, v in train])
    X = Xy[:, :2]
    y = Xy[:, -1]

    accuracy_on_test_set = sum(y == p.predict(x) for x, y in zip(X, y))
    accuracy_on_test_set /= float(len(train))

    return (accuracy_on_training_set, accuracy_on_test_set)
