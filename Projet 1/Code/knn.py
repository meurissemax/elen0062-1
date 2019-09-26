"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data import make_data1, make_data2
from plot import plot_boundary


if __name__ == "__main__":
    # General variables
    datasets_functions = [make_data1, make_data2]
    n_samples = 2000
    n_training = 150
    n_neighbors = [1, 5, 10, 75, 100, 150]

    seed = 13091006  # a fixed seed to have reproducible experiments
    p_test = (n_samples - n_training) / n_samples

    # Generate datasets
    random = np.random.seed(seed)
    datasets = list()

    for f in datasets_functions:
        datasets.append(f(n_samples, random))

    ##############
    # Question 1 #
    ##############

    # Apply the algorithm on each dataset
    for count, dataset in enumerate(datasets):

        # Get training and testing sets
        X = dataset[0]
        y = dataset[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_test)

        # K Neighbors algorithm
        for n in n_neighbors:
            neigh = KNeighborsClassifier(n_neighbors=n)
            neigh.fit(X_train, y_train)  # train the algorithm

            # Save results
            fig_name = 'figures/knn/dataset' + str(count + 1) + '/' + str(n)

            plot_boundary(fig_name, neigh, X_test, y_test)

    ##############
    # Question 2 #
    ##############

    # Variables
    k = 10
    accuracies = np.zeros((len(datasets), k, len(n_neighbors)))

    # Apply the algorithm with k-fold cross validation on each dataset
    for count, dataset in enumerate(datasets):
        kf = KFold(n_splits=k)

        X = dataset[0]
        y = dataset[1]

        # Try algorithm with each training and testing subsets
        for i, [train_index, test_index] in enumerate(kf.split(X)):

            # Get training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # K Neighbors algorithm
            for j, n in enumerate(n_neighbors):
                neigh = KNeighborsClassifier(n_neighbors=n)
                neigh.fit(X_train, y_train)  # train the algorithm

                # Get and save the score
                accuracies[count][i][j] = neigh.score(X_test, y_test)

    # Show results
    print(accuracies)
