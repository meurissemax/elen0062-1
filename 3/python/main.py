"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 3 - Activity prediction for chemical compounds

Authors :
    - Maxime Meurisse
    - Francois Rozet
    - Valentin Vermeylen
"""

#############
# Libraries #
#############

import sys
import utils

sys.path.insert(1, '../models')

from knn import Model


#####################
# General variables #
#####################

TRAINING_SET = '../resources/csv/training_set.csv'
TEST_SET = '../resources/csv/test_set.csv'


########
# Main #
########

if __name__ == '__main__':
    # Load training and testing data
    LS = utils.load_from_csv(TRAINING_SET)
    TS = utils.load_from_csv(TEST_SET)

    #########
    # Model #
    #########

    # LEARNING

    # Create fingerprint features and output
    X_LS = utils.create_fingerprints(LS['SMILES'].values)
    y_LS = LS['ACTIVE'].values

    # Build and train the model
    model = Model()
    model.train(X_LS, y_LS)

    # PREDICTION

    X_TS = utils.create_fingerprints(TS['SMILES'].values)

    # Predict
    y_pred = model.get_pred(X_TS)

    # Estimated AUC of the model
    auc_predicted = 0.50

    # Making the submission file
    fname = utils.make_submission(y_pred, auc_predicted, '../products/submission')

    print('Submission file "{}" successfully written'.format(fname))
