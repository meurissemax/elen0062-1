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

from matplotlib import rc
from matplotlib import pyplot as plt


#################
# Configuration #
#################

font = {'size': 14}

rc('font', **font)
rc('text', usetex=True)


#############
# Functions #
#############

def scatter_plot(x, y, fig_name):
    """Create a scatter plot

    Inputs
    ------
    x : array of shape [n]
    y : array of shape [n]
    fig_name : str
        the name of the saved figure
        (without extension)
    """

    plt.figure()

    try:
        plt.xlabel('$x_r$')
        plt.ylabel('$y$')
        plt.grid(True, linewidth=.2)

        plt.scatter(x, y, s=.2)

        plt.savefig('../products/{}.pdf'.format(fig_name))
    finally:
        plt.close()


def multi_plot(x, x_label, Y, legends, fig_name, x_log=False, y_log=True, y_lim=False):
    """Create a multi line plot

    Inputs
    ------
    x : array of shape [n]
    x_label : str
        the x label of the plot
    Y : array of shape [k, n]
    legends : array of shape [k]
        legend of each y vector
    fig_name : str
        the name of the saved figure
        (without extension)
    x_log : boolean
        to decide if the scale of the
        x-axis is log or not
        (default = False)
    y_log : boolean
        to decide if the scale of the
        y-axis is log or not
        (default = True)
    y_lim : boolean
        to decide if the y-axis is
        limited to values
        (default = False)
    """

    plt.figure()

    try:
        plt.xlabel(x_label)

        if x_log is True:
            plt.xscale('log')

        if y_log is True:
            plt.yscale('log')

        if y_lim is True:
            plt.ylim(0.000001, 1)

        plt.grid(True, linewidth=.2)

        for y in Y:
            plt.plot(x, y, linewidth=2)

        plt.legend(legends)

        plt.savefig('../products/{}.pdf'.format(fig_name))
    finally:
        plt.close()
