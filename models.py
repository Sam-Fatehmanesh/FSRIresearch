import sklearn
from sklearn import linear_model
import torch
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import datasets
import random
import numpy as np


class Ensemble:
    def __init__(self, model, model_count, seed=42):
        # generates each new sklearn model with a new seed
        self.models = []
        for i in range(model_count):
            np.random.seed(seed + i)
            self.models.append(model())

    def fit(self, X, Y):
        for i in range(len(self.models)):
            self.models[i].fit(X, Y)
    
    def raw_inf(self, X):
        return [model.predict(X) for model in self.models]

    # returns mean and variance of the mean
    def predict(self, X):
        outputs = self.raw_inf(X)
        mean = sum(self.raw_inf(X)) / len(self.models)
        return mean, sum([(outputs[i] - mean)**2 for i in range(len(self.models))]) / len(self.models)

    def printParams(self):
        for i in self.models:
            print(i.get_params())
    

def get_linear_ensemble(model_count):
    return Ensemble(linear_model.LinearRegression, model_count)

def get_svm_ensemble(model_count):
    return Ensemble(svm.SVC, model_count)

def get_GP_ensemble(model_count):
    return Ensemble(GaussianProcessRegressor, model_count)


diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

m = get_linear_ensemble(4)

#m.fit(diabetes_X, diabetes_y)
#print(diabetes_X[0])
# print(diabetes_y[0])
# print(m.raw_inf([diabetes_X[0]]))
m.printParams()