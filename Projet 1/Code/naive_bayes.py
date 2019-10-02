"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split

from scipy.stats import norm

from data import make_data1, make_data2
from plot import plot_boundary


class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        """Fit a Gaussian naive Bayes model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Get classes and data information
        classes, indexes = np.unique(y, return_inverse=True)

        n_classes = len(classes)
        n_attributes = len(X[0])
        n_data = len(y)

        # Calculate prior probabilities
        class_prior_ = np.zeros((n_classes))

        for i, classe in enumerate(classes):
            class_prior_[i] = len(y[y == classe]) / n_data

        # Calculate mean and variance for each attributes, grouped by classes
        theta_ = np.zeros((n_classes, n_attributes))
        sigma_ = np.zeros((n_classes, n_attributes))

        for i, classe in enumerate(classes):
            for j in range(n_attributes):
                theta_[i][j] = np.mean(X[y == classe, j])
                sigma_[i][j] = np.var(X[y == classe, j])

        # Define instance variables
        self.class_prior_ = class_prior_
        self.theta_ = theta_
        self.sigma_ = sigma_

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # Check if estimator has been fitted
        check_is_fitted(self, ['class_prior_', 'theta_', 'sigma_'])

        # Input validation
        X = check_array(X)

        # Predict classe of each point
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # Check if estimator has been fitted
        check_is_fitted(self, ['class_prior_', 'theta_', 'sigma_'])

        # Input validation
        X = check_array(X)

        # Predict probabilities of each classe for each point
        p = np.zeros((len(X), len(self.class_prior_)))

        for i, attributes in enumerate(X):
            p_classes = self.class_prior_.copy()

            for j in range(len(self.class_prior_)):
                for k, attribute in enumerate(attributes):
                    p_classes[j] *= norm.pdf(attribute, self.theta_[j][k], np.sqrt(self.sigma_[j][k]))

            p[i] = [p_classes[i] / sum(p_classes) for i in range(len(p_classes))]

        return p


if __name__ == "__main__":
    # General variables
    datasets = [make_data1, make_data2]
    n_samples, p_test = 2000, 0.925

    ##############
    # Question 3 #
    ##############

    # Variables
    n_show = int(0.25 * (p_test * n_samples))
    accuracies = np.zeros((len(datasets)))

    # Apply the algorithm on each dataset
    for i, f in enumerate(datasets):

        # Get training and testing sets
        X, y = f(n_samples, random_state=0)  # seed fixed to 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_test, shuffle=False)

        # Gaussian Naive Bayes algorithm
        gnb = GaussianNaiveBayes()
        gnb.fit(X_train, y_train)  # train the algorithm

        # Get accuracies
        accuracies[i] = gnb.score(X_test, y_test)

        # Save results
        fig_name = f.__name__ + '_gnb'
        plot_boundary(fig_name, gnb, X_test[:n_show], y_test[:n_show])

    # Display accuracies
    for accuracy in accuracies:
        print('accuracy of ' + str(accuracy))
