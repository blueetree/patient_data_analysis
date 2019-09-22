import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from fancyimpute import IterativeImputer
from sklearn.impute import SimpleImputer


class DFIterativeImputer(BaseEstimator, TransformerMixin):

    def __init__(self, max_iter=10):
        self.imputer = None
        self.max_iter = max_iter

    def fit(self, X, y=None):
        self.imputer = IterativeImputer(max_iter=self.max_iter)
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_filled = self.imputer.transform(X)
        X_filled = pd.DataFrame(X_filled, index=X.index, columns=X.columns)
        return X_filled


class DFSimpleImputer(BaseEstimator, TransformerMixin):

    def __init__(self, max_iter=10):
        self.imputer = None

    def fit(self, X, y=None):
        self.imputer = SimpleImputer()
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_filled = self.imputer.transform(X)
        X_filled = pd.DataFrame(X_filled, index=X.index, columns=X.columns)
        return X_filled


class DFStandardScaler(TransformerMixin):

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled







