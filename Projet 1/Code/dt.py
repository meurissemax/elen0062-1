"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data import make_data1, make_data2
from plot import plot_boundary


def generate_datasets(datasets_functions, n_samples, seed):
    """ Generate random datasets

    Parameters
    ----------
    datasets_functions : 1-d array of functions
        functions used to generate datasets
    n_samples : int > 0
        the number of samples in each dataset
    seed : int > 0
        the seed used to generate random data

    Return
    ------
    datasets : 1-d array of datasets
        array of generated datasets
    """

    random = np.random.seed(seed)
    datasets = list()

    for f in datasets_functions:
        datasets.append(f(n_samples, random))

    return datasets


def bar_chart(datasets, title, x_label, y_label, x_legend, fig_name):
    """ Generate a bar plot of data about n datasets

    Parameters
    ----------
    datasets : a 1-d array with n datasets
        the list of datasets to plot (the size of each
        dataset must be s)
    title : str
        the title of the plot
    x_label : str
        the x label of the plot
    y_label : str
        the y label of the plot
    x_legend : a 1-d array of str (len(x_legend) = s)
        the x legend of the plot
    fig_name : str
        the name of the figure to export
    """

    n_datasets = len(datasets)
    bar_width = (1 / n_datasets) - 0.1

    # Set positions of bars on x axis
    r = list()
    r.append(np.arange(len(x_legend)))

    for i in range(n_datasets - 1):
        r.append([x + bar_width for x in r[-1]])

    # Create figure
    fig, ax = plt.subplots()

    # Add the bars
    for i, dataset in enumerate(datasets):
        ax.bar(r[i], dataset, bar_width, label='Dataset ' + str(i + 1))

    # Add title
    ax.set_title(title)

    # Add axis label
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Axis styling.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)

    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add legend
    ax.set_xticks(r[0] + (n_datasets - 1) * (bar_width / 2))
    ax.set_xticklabels(x_legend)

    ax.legend()

    # Generate and save figure
    fig.tight_layout()
    fig.savefig(fig_name)


if __name__ == "__main__":
    # General variables
    datasets_functions = [make_data1, make_data2]
    n_samples = 2000
    n_training = 150
    max_depth = [1, 2, 4, 8, None]

    seed = 13091006  # a fixed seed to have reproducible experiments
    p_test = (n_samples - n_training) / n_samples

    ##############
    # Question 1 #
    ##############

    # Generate datasets
    datasets = generate_datasets(datasets_functions, n_samples, seed)

    # Apply the algorithm on each dataset
    for count, dataset in enumerate(datasets):

        # Get training and testing sets
        X = dataset[0]
        y = dataset[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_test)

        # Decision Tree algorithm
        for depth in max_depth:
            dt = DecisionTreeClassifier(max_depth=depth)
            dt.fit(X_train, y_train)  # train the algorithm

            # Save results
            fig_name = 'figures/dt/dataset' + str(count + 1) + '/' + str(depth)

            plot_boundary(fig_name, dt, X_test, y_test)

    ##############
    # Question 2 #
    ##############

    # Variables
    n_generations = 5

    n_datasets = len(datasets_functions)
    n_depths = len(max_depth)

    accuracies = np.zeros((n_generations, n_datasets, n_depths))
    mean = np.zeros((n_datasets, n_depths))
    std = np.zeros((n_datasets, n_depths))

    # Calculate accuracies for multiple generations
    for generation in range(n_generations):

        # Generate datasets
        datasets = generate_datasets(datasets_functions, n_samples, np.random.randint(1, 1000))

        # Apply the algorithm on each dataset
        for i, dataset in enumerate(datasets):

            # Get training and testing sets
            X = dataset[0]
            y = dataset[1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_test)

            # Decision Tree algorithm
            for j, depth in enumerate(max_depth):
                dt = DecisionTreeClassifier(max_depth=depth)
                dt.fit(X_train, y_train)  # train the algorithm

                # Get and save the score
                accuracies[generation][i][j] = dt.score(X_test, y_test)

    # Calculate the mean and standard deviation over the multiple generations
    for dataset in range(n_datasets):
        for depth in range(n_depths):

            # We get the accuracies over the multiple generations
            # for fixed dataset and depth
            e = np.zeros(n_generations)

            for generation in range(n_generations):
                e[generation] = accuracies[generation][dataset][depth]

            # We calculate mean and standard deviation
            mean[dataset][depth] = np.mean(e)
            std[dataset][depth] = np.std(e)

    # Plot and save results
    x_legend = list(map(str, max_depth))

    bar_chart(mean, 'Average test set accuracies', 'Depth of the decision tree', 'Accuracy', x_legend, 'figures/dt/accuracy/mean.pdf')
    bar_chart(std, 'Standard deviation of each test set accuracies', 'Depth of the decision tree', 'Standard deviation', x_legend, 'figures/dt/accuracy/std.pdf')
