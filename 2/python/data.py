"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 2 - Bias and variance analysis

Authors :
    - Maxime Meurisse
    - Valentin Vermeylen
"""

#############
# Libraries #
#############

import numpy as np

from sklearn.utils import check_random_state


#############
# Functions #
#############

def make_data(n_samples, n_irrelevant=0, random_state=None):
    """Generate random samples (partially
    adapted from project 1's data generation
    function)

    Inputs
    ------
    n_samples : int > 0
        The number of samples to generate.
    n_irrelevant : int >= 0
        The number of irrelevant feature to add
        (default = 0).
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Outputs
    -------
    X : array of shape [n_samples, 1 + n_irrelevant]
    y : array of shape [n_samples]
    """
    drawer = check_random_state(random_state)

    X = drawer.uniform(-10, 10, (n_samples, 1 + n_irrelevant))
    X = np.round(X, decimals=1)

    noise = 0.1 * drawer.normal(0, 1, n_samples)

    y = np.sin(X[:, 0]) * np.exp(-((X[:, 0] ** 2) / 16)) + noise

    return X, y
