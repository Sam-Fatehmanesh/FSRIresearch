import sklearn
from sklearn import linear_model
import torch
from sklearn import svm



class Ensemble:
    def __init__(self, model, model_count, seed=42):
        self.models = [model() for i in range(model_count)]

    def fit(self, X, Y):
        for i in range(len(self.models)):
            self.models[i].fit(X, Y)
    
    def raw_inf(self, X):
        return [model.predict(X) for model in self.models]

    # returns mean, and variance of the mean
    def predict(self, X):
        return sum(self.raw_inf(X)) / len(self.models), sum([(self.raw_inf(X)[i] - self.predict(X)[0])**2 for i in range(len(self.models))]) / len(self.models)
    

def get_linear_ensemble(model_count, seed=42):
    return Ensemble(linear_model.LinearRegression, model_count)

def get_svm_ensemble(model_count, seed=42):
    return Ensemble(svm.SVC, model_count)

