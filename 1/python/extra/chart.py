"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc

rc('text', usetex=True)


def line_chart(x, y, x_label, y_label, fig_name):
    # Create figure
    fig, ax = plt.subplots()

    # Add the line
    ax.plot(x, y)

    # Add the maximum point
    y_max = max(y)
    x_max = x[np.argmax(y)]

    ax.plot(x_max, y_max, 'ro')

    # Add axis label
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Axis styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)

    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(True, color='#EEEEEE')

    # Generate and save figure
    fig.tight_layout()
    fig.savefig(fig_name)


def bar_chart(datasets, x_label, y_label, x_legend, fig_name):
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
