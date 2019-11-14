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

from data import make_data
from protocol import Protocol
from plot import scatter_plot
from plot import multi_plot

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


#####################
# General variables #
#####################

# Number of datasets to generate
N_DATASETS = 10

# Number of samples per dataset
N_SAMPLES = 1000

# Number of irrelevant features
N_IRRELEVANT = 0

# Models to use
MODELS = [Ridge, KNeighborsRegressor]

# Model complexity
COMPLEXITIES = [1.0, 5]


########
# Main #
########

if __name__ == '__main__':
    # Generate data
    X, y = make_data(N_DATASETS * N_SAMPLES, N_IRRELEVANT)

    # Plot data
    scatter_plot(X[:, 0], y, 'Q3d_data')

    # Calculate expected error and its terms for
    # each model
    for model, complexity in zip(MODELS, COMPLEXITIES):
        # Create the protocol
        p = Protocol(X, y)

        # Train models
        p.train(model, complexity, N_DATASETS)

        # Get error and its terms
        noise, s_bias, var, exp_error = p.eval()

        # Plot result
        multi_plot(
            p.get_unique(),
            '$x_r$',
            [noise, s_bias, var, exp_error],
            ['residual error', 'squared bias', 'variance', 'expected error'],
            'Q3d_' + model.__name__
        )
