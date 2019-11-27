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

from sklearn.tree import DecisionTreeClassifier


#########
# Class #
#########

class Model:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_LS, y_LS):
        self.model.fit(X_LS, y_LS)

    def get_pred(self, X_TS):
        return self.model.predict_proba(X_TS)[:, 1]
