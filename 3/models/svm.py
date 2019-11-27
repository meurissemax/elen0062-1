"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 3 - Activity prediction for chemical compounds

Authors :
    - Maxime Meurisse
    - François Rozet
    - Valentin Vermeylen
"""

#############
# Libraries #
#############

from sklearn.svm import SVC


#########
# Class #
#########

class Model:
    def __init__(self):
        self.model = SVC(gamma='scale', probability=True)

    def train(self, X_LS, y_LS):
        self.model.fit(X_LS, y_LS)

    def get_pred(self, X_TS):
        return self.model.predict_proba(X_TS)[:, 1]
