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

from knn import Model as knn
from lda import Model as lda
from mlp import Model as mlp
from svm import Model as svm


#####################
# General variables #
#####################

TRAINING_SET = '../resources/csv/training_set.csv'
TEST_SET = '../resources/csv/test_set.csv'

MODELS = [knn, lda, mlp, svm]


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

    # Build and train models
    models = list()

    for model in MODELS:
        m = model()
        m.train(X_LS, y_LS)
        
        models.append(m)

    # PREDICTION

    X_TS = utils.create_fingerprints(TS['SMILES'].values)

    # Predict for each model
    for i, model in enumerate(models):
    	if i == 0:
    		y_pred = model.get_pred(X_TS)
    	else:
        	y_pred += model.get_pred(X_TS)

    # Get the mean of all predictions
    y_pred /= len(models)

    # Estimated AUC of the model
    auc_predicted = 0.50

    # Making the submission file
    fname = utils.make_submission(y_pred, auc_predicted, '../products/submission')

    print('Submission file "{}" successfully written'.format(fname))
