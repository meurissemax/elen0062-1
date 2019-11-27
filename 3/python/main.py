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

import os
import utils

from models import KNN
from models import LDA
from models import MLP
from models import SVM
from models import MeanClassifier


##############
# Parameters #
##############

TRAINING_SET = '../resources/csv/training_set.csv'
TEST_SET = '../resources/csv/test_set.csv'

DESTINATION = '../products/'

MODEL = MeanClassifier([KNN(n_neighbors=15), MLP(), SVM()])


########
# Main #
########

if __name__ == '__main__':
    # Load training and test set
    LS = utils.load_from_csv(TRAINING_SET)
    TS = utils.load_from_csv(TEST_SET)

    # Create fingerprint features and output of learning set
    X_LS = utils.create_fingerprints(LS['SMILES'].values)
    y_LS = LS['ACTIVE'].values

    # Train model
    MODEL.fit(X_LS, y_LS)

    # Create fingerprint features of test set
    X_TS = utils.create_fingerprints(TS['SMILES'].values)

    # Predict
    prob = MODEL.active_proba(X_TS)

    # Estimated AUC of the model
    auc_predicted = 0.50

    # Writing the submission file
    os.makedirs(DESTINATION, exist_ok=True)
    fname = utils.make_submission(prob, auc_predicted, DESTINATION + 'submission')

    print('Submission file "{}" successfully written'.format(fname))
