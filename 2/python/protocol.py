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

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


###########
# Classes #
###########

class Protocol:
    def __init__(self, X, y):
        """Constructor of the class. Save
        X and y vectors.

        Inputs
        ------
        X : array of shape [n_samples, n_features]
        y : array of shape [n_samples]
        """

        self.X = X
        self.y = y

    def get_unique(self):
        """ Return unique feature of X.

        Outputs
        -------
        array with unique feature of X
        """

        return np.unique(self.X, axis=0)

    def train(self, model, complexity, n_datasets):
        """Trains several models on datasets

        Inputs
        ------
        model : class of the model to train
        complexity : complexity of the model
        n_datasets : number of datasets to use
            and so number of models to train
        """

        self.models = list()

        # Split dataset into multiple training sets
        datasets = zip(
            np.split(self.X, n_datasets),
            np.split(self.y, n_datasets)
        )

        # Train each model with one of the training set
        for X_train, y_train in datasets:
            if model == Ridge:
                instance = model(alpha=complexity)
            elif model == KNeighborsRegressor:
                instance = model(n_neighbors=complexity)
            else:
                instance = model()

            self.models.append(instance.fit(X_train, y_train))

    def eval(self):
        """Returns the error and its terms

        Outputs
        -------
        noise : array of shape [n_unique_samples]
            The noise term.
        s_bias : array of shape [n_unique_samples]
            The squared bias term.
        var : array of shape [n_unique_samples]
            The variance term.
        exp_error : array of shape [n_unique_samples]
            The expected error, i.e the sum of
            the noise, the squared bias and
            the variance
        """
        X_unique = self.get_unique()

        n = (len(X_unique), 1)

        noise = np.zeros(n)
        s_bias = np.zeros(n)
        var = np.zeros(n)

        # Iterate for each unique value of feature
        for i, x_i in enumerate(X_unique):
            # Get y values related to the x_i value
            indexes = self.X == x_i
            y_rel = self.y[indexes[:, 0]]

            # Get the y values predicted by models for
            # the x_i value
            y_pred = [m.predict(x_i.reshape(1, -1)) for m in self.models]

            # Calculate the 3 terms
            noise[i] = np.var(y_rel)
            s_bias[i] = np.square(np.mean(y_rel) - np.mean(y_pred))
            var[i] = np.var(y_pred)

        # Calculate the expected error for each unique
        # value of feature
        exp_error = noise + s_bias + var

        return noise, s_bias, var, exp_error
