# -*- coding: utf-8 -*-
from itertools import groupby


def feature1(x):
    """This feature computes the proportion of black squares to the
       total number of squares in the grid.
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature1_value: type-float
       """
    return sum(sum(i) for i in x) / float(len(x) * len(x[0]))


def feature2(x):
    """This feature computes the sum of the max of continuous black squares
       in each row
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature2_value: type-float
       """
    return sum(max(len(list(v)) * k for k, v in groupby(z)) for z in x)
