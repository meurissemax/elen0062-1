"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from data import make_data1, make_data2
from plot import plot_boundary

# from extra.chart import line_chart


if __name__ == "__main__":
    # General variables
    datasets = [make_data1, make_data2]
    n_samples, p_test = 2000, 0.925
    n_neighbors = [1, 5, 10, 75, 100, 150]

    ##############
    # Question 1 #
    ##############

    # Variables
    n_show = int(0.25 * (p_test * n_samples))

    # Apply the algorithm on each dataset
    for f in datasets:

        # Get training and testing sets
        X, y = f(n_samples, random_state=0)  # seed fixed to 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_test, shuffle=False)

        # K Neighbors algorithm
        for n in n_neighbors:
            estimator = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)

            # Save results
            fig_name = f.__name__ + '_knn_' + str(n)
            plot_boundary(fig_name, estimator, X_test[:n_show], y_test[:n_show])

    ##############
    # Question 2 #
    ##############

    # Variables
    k, k_neighbors = 10, range(5, 150, 1)  # ten-fold cross validation
    accuracies_mean = np.zeros((len(k_neighbors)))

    # Get the second dataset
    X, y = datasets[1](n_samples, random_state=0)  # seed fixed to 0

    # Apply the algorithm with k-fold cross validation on second dataset
    for i, n in enumerate(k_neighbors):
        neigh = KNeighborsClassifier(n_neighbors=n)
        accuracies = cross_val_score(neigh, X, y, cv=k)
        accuracies_mean[i] = np.mean(accuracies)

    # Get the optimal value of n_neighbors
    n_optimal = k_neighbors[np.argmax(accuracies_mean)]

    # Display results
    print('(dataset 2) optimal number of neighbors : ' + str(n_optimal) + ' with a score of ' + str(max(accuracies_mean)))

    """
    # Plot results
    x_label = 'number of neighbors'
    y_label = 'accuracy'
    fig_name = 'outputs/knn_kfold_scores.pdf'

    line_chart(k_neighbors, accuracies_mean, x_label, y_label, fig_name)
    """
