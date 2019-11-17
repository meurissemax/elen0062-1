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

from data import make_data
from protocol import Protocol
from plot import multi_plot

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


#####################
# General variables #
#####################

# Number of datasets to generate
N_DATASETS = 10

# Models to use
MODELS = [Ridge, KNeighborsRegressor]


########
# Main #
########

if __name__ == '__main__':
    ############################
    # Size of the learning set #
    ############################

    # Variable element
    n_samples = range(10, 3500, 100)

    # Fixed elements
    complexities = [1.0, 5]
    n_irrelevants = 0

    # Calculate values for each model
    for model, complexity in zip(MODELS, complexities):
        m_noise, m_s_bias = list(), list()
        m_var, m_exp_error = list(), list()

        for n_sample in n_samples:
            # Generate data
            X, y = make_data(N_DATASETS * int(n_sample), n_irrelevants)

            # Create the protocol
            p = Protocol(X, y)

            # Train models
            p.train(model, complexity, N_DATASETS)

            # Get error and its terms
            noise, s_bias, var, exp_error = p.eval()

            # Save mean values of each term
            m_noise.append(np.mean(noise))
            m_s_bias.append(np.mean(s_bias))
            m_var.append(np.mean(var))
            m_exp_error.append(np.mean(exp_error))

        # Plot result
        multi_plot(
            n_samples,
            'size of the learning set',
            [m_noise, m_s_bias, m_var, m_exp_error],
            [
                'mean residual error',
                'mean squared bias',
                'mean variance',
                'mean expected error'
            ],
            'Q3e_' + model.__name__ + '_size_ls',
            x_log=False,
            y_log=True,
            y_lim=True
        )

    ####################
    # Model complexity #
    ####################

    # Variable element
    complexities = [np.logspace(-5, 1, num=5, base=10.0), range(5, 1000, 20)]

    # Fixed element
    n_samples = 1000
    n_irrelevants = 0

    # Plot parameter
    x_log = [True, False]

    # Calculate values for each model
    for i, model in enumerate(MODELS):
        m_noise, m_s_bias = list(), list()
        m_var, m_exp_error = list(), list()

        for complexity in complexities[i]:
            # Generate data
            X, y = make_data(N_DATASETS * n_samples, n_irrelevants)

            # Create the protocol
            p = Protocol(X, y)

            # Train models
            p.train(model, complexity, N_DATASETS)

            # Get error and its terms
            noise, s_bias, var, exp_error = p.eval()

            # Save mean values of each term
            m_noise.append(np.mean(noise))
            m_s_bias.append(np.mean(s_bias))
            m_var.append(np.mean(var))
            m_exp_error.append(np.mean(exp_error))

        # Plot result
        multi_plot(
            complexities[i],
            'model complexity',
            [m_noise, m_s_bias, m_var, m_exp_error],
            [
                'mean residual error',
                'mean squared bias',
                'mean variance',
                'mean expected error'
            ],
            'Q3e_' + model.__name__ + '_model_complexity',
            x_log=x_log[i],
            y_log=True,
            y_lim=True
        )

    ########################
    # Irrelevant variables #
    ########################

    # Variable element
    n_irrelevants = range(0, 6, 1)

    # Fixed elements
    n_samples = 1000
    complexities = [1.0, 5]

    # Calculate values for each model
    for model, complexity in zip(MODELS, complexities):
        m_noise, m_s_bias = list(), list()
        m_var, m_exp_error = list(), list()

        for n_irrelevant in n_irrelevants:
            # Generate data
            X, y = make_data(N_DATASETS * n_samples, n_irrelevant)

            # Create the protocol
            p = Protocol(X, y)

            # Train models
            p.train(model, complexity, N_DATASETS)

            # Get error and its terms
            noise, s_bias, var, exp_error = p.eval()

            # Save mean values of each term
            m_noise.append(np.mean(noise))
            m_s_bias.append(np.mean(s_bias))
            m_var.append(np.mean(var))
            m_exp_error.append(np.mean(exp_error))

        # Plot result
        multi_plot(
            n_irrelevants,
            'number of irrelevant variables',
            [m_noise, m_s_bias, m_var, m_exp_error],
            [
                'mean residual error',
                'mean squared bias',
                'mean variance',
                'mean expected error'
            ],
            'Q3e_' + model.__name__ + '_irrelevant_var',
            x_log=False,
            y_log=True,
            y_lim=True
        )
