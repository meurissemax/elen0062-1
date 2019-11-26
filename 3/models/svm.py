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
    def get_model(self):
        model = SVC(gamma='scale', probability=True)

        return model
