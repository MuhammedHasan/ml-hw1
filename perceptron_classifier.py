# -*- coding: utf-8 -*-

import numpy as np
import random
SEED = 0
random.seed(SEED)


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.
            Parameters
            ----------
            X : {array-like}, shape = [n_samples, n_features]
            n_features is the number of features.
            y : array-like, shape = [n_samples]
            Target values.
            Returns
            -------
            self : object
        """

        # import pdb
        # pdb.set_trace()

        current_iter = 0.0
        total_iter = self.n_iter * X.shape[0]

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            Xy = zip(X, y)
            np.random.shuffle(Xy)

            for xi, target in Xy:
                update = self.eta * (target - self.predict(xi))
                update *= (total_iter - current_iter) / total_iter
                current_iter += 1

                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
