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

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


###########
# Classes #
###########

class ActivityClassifier():
    def active_proba(self, X):
        return self.predict_proba(X)[:, -1]

class KNN(KNeighborsClassifier, ActivityClassifier):
    pass

class LDA(LinearDiscriminantAnalysis, ActivityClassifier):
    pass

class MLP(MLPClassifier, ActivityClassifier):
    pass

class SVM(SVC, ActivityClassifier):
    @staticmethod
    def kernel(X, Y):
        
        pass

    def __init__(self, kernel='linear', probability=True, *args, **kwargs):
        super().__init__(kernel=kernel, probability=probability, *args, **kwargs)

class MeanClassifier(ActivityClassifier):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

        return self

    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], 2))

        for model in self.models:
            proba += model.predict_proba(X)

        proba /= len(self.models)

        return proba
