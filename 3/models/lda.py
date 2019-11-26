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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#########
# Class #
#########

class Model:
    def get_model(self):
        model = LinearDiscriminantAnalysis()

        return model
