# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cPickle as pickle
from perceptron_classifier import *

from feature_extraction import *


def read_featurize_set(set_name="training"):
    train_positives = pickle.load(open(set_name + '_set_positives.p', 'rb'))
    train_negatives = pickle.load(open(set_name + '_set_negatives.p', 'rb'))

    train_positives = [(1, x) for _, x in train_positives.items()]
    train_negatives = [(-1, x) for _, x in train_negatives.items()]

    train = train_positives + train_negatives

    Xy = np.array([[feature1(v), feature2(v), k] for k, v in train])
    X = Xy[:, :2]
    y = Xy[:, -1]

    return (X, y)


def visualise_features():
    """This function should generate a scatterplot for all the training
       examples using the features you defined before
       """

    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))

    plt.scatter([feature1(v) for _, v in train_positives.items()],
                [feature2(v) for _, v in train_positives.items()],
                color='blue', marker='x', label='positives')

    plt.scatter([feature1(v) for _, v in train_negatives.items()],
                [feature2(v) for _, v in train_negatives.items()],
                color='red', marker='o', label='negatives')

    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    return


def train_classifier():
    """This function is for training the classifier and learning the weights
       for the decision boundary
       Returns
       -------
       weights: from the perceptron classifier
       """
    X, y = read_featurize_set()

    p = Perceptron(eta=0.1, n_iter=1000)
    p.fit(X, y)

    return p.w_


def visualize_decision_boundary():
    """Using the weights learnt by your classifier, a call to this function
       should plot the decision boundary over the dataser"""

    X, y = read_featurize_set()

    ppn = Perceptron(eta=0.1, n_iter=1000)
    ppn.fit(X, y)

    def plot_decision_regions(X, y, classifier, resolution=0.001):

        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    return
