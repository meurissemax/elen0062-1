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

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


###########
# Classes #
###########

class KNN(KNeighborsClassifier):
    pass

class LDA(LinearDiscriminantAnalysis):
    pass

class MLP(MLPClassifier):
    pass

class RFC(RandomForestClassifier):
    pass

class SVM(SVC):
    def __init__(self, kernel='rbf', probability=True, gamma='scale', C=1):
        super().__init__(kernel=kernel, probability=probability, gamma=gamma, C=C)       

class MeanClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None, geometric=False):
        if weights is None:
            self.weights = [1] * len(models)
        else:
            self.weights = weights

        self.models = models

        self.geometric = geometric

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

        return self

    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], 2))
        if self.geometric:
            proba += 1

        for i, model in enumerate(self.models):
            temp = model.predict_proba(X)
            if self.geometric:
                proba *= temp ** self.weights[i]
            else:
                proba += temp * self.weights[i]

        n = sum(self.weights)
        if self.geometric:
            proba = proba ** (1 / n)
        else:
            proba /= n

        return proba

    def predict(self, X):
        y = np.argmax(self.predict_proba(X), axis=1)
        return y
