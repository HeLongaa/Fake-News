# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def try_divide(x, y):
    """
        Try to divide two numbers
    """
    val = 0.0
    if y != 0.0:
        val = float(x) / float(y)
    return val


def cosine_sim(x, y):
    d = 0.0
    try:
        if isinstance(x, np.ndarray):
            x = x.reshape(1, -1)  # get rid of the warning
        if isinstance(y, np.ndarray):
            y = y.reshape(1, -1)
        d = cosine_similarity(x, y)
        d = d[0][0]
    except Exception as e:
        print("x,y:", x, y, e)
    return d
