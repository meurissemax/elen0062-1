"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms

Authors :
    - Maxime Meurisse
    - Valentin Vermeylen
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from data import make_data1, make_data2
from plot import plot_boundary

# from extra.chart import bar_chart


if __name__ == "__main__":
    # General variables
    datasets = [make_data1, make_data2]
    n_samples, p_test = 2000, 0.925  # p_test : proportion of testing points
    max_depth = [1, 2, 4, 8, None]

    ##############
    # Question 1 #
    ##############

    # Variables
    n_show = int(0.25 * (p_test * n_samples))  # number of points to plot

    # Apply the algorithm on each dataset
    for f in datasets:

        # Get training and testing sets
        X, y = f(n_samples, random_state=0)  # seed fixed to 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_test, shuffle=False)

        # Decision Tree algorithm
        for depth in max_depth:
            estimator = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)

            # Save results
            fig_name = f.__name__ + '_dt_' + str(depth)
            plot_boundary(fig_name, estimator, X_test[:n_show], y_test[:n_show])

    ##############
    # Question 2 #
    ##############

    # Variables
    n_generations, n_datasets, n_depths = 5, len(datasets), len(max_depth)
    accuracies = np.zeros((n_generations, n_datasets, n_depths))

    # Calculate accuracies for multiple generations
    for generation in range(n_generations):

        # Apply the algorithm on each dataset
        for i, f in enumerate(datasets):

            # Get training and testing sets
            X, y = f(n_samples, random_state=generation)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_test, shuffle=False)

            # Decision Tree algorithm
            for j, depth in enumerate(max_depth):
                dt = DecisionTreeClassifier(max_depth=depth)
                dt.fit(X_train, y_train)  # train the algorithm

                # Get and save the score
                accuracies[generation][i][j] = dt.score(X_test, y_test)

    # Calculate the mean and standard deviation over the multiple generations
    accuracies_mean = np.mean(accuracies, 0)
    accuracies_std = np.std(accuracies, 0)

    # Display results
    print("accuracies_mean")
    print(accuracies_mean)

    print("accuracies_std")
    print(accuracies_std)

    """
    # Plot results
    x_label = 'depth'
    x_legend = ['$' + str(depth) + '$' for depth in max_depth]

    bar_chart(accuracies_mean, x_label, 'mean', x_legend, 'outputs/dt_accuracies_mean.pdf')
    bar_chart(accuracies_std, x_label, 'standard deviation', x_legend, 'outputs/dt_accuracies_std.pdf')
    """
