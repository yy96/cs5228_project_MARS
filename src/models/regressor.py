from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
     mean_squared_error
)

from catboost import Pool


class Regressor(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnRegressor():
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.model = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.model.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):
        prediction = self.predict(df_test)
        actual = df_test[self.target]

        return compute_metrics_collection(actual, prediction)

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df[self.features].values)
    
    def get_params(self):
        return self.model.get_params()


class CategoricalBoostRegressor():
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str, cat_features_loc: List[int]
    ):
        self.model = estimator
        self.features = features
        self.target = target
        self.cat_features_loc = cat_features_loc

    def _create_data_pool(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        train_x = df_train[self.features]
        train_y = df_train[self.target]
        train_pool = Pool(data=train_x, label=train_y, cat_features=self.cat_features_loc)

        test_x = df_test[self.features]
        test_y = df_test[self.target]
        test_pool = Pool(data=test_x, label=test_y, cat_features=self.cat_features_loc)
        return train_pool, test_pool

    def train(self, df_train: pd.DataFrame):
        train_x = df_train[self.features]
        train_y = df_train[self.target]
        train_pool = Pool(data=train_x, label=train_y, cat_features=self.cat_features_loc)
        self.model.fit(train_pool)
            
    def evaluate(self, df_test: pd.DataFrame):
        prediction = self.predict(df_test)
        actual = df_test[self.target]

        return compute_metrics_collection(actual, prediction)

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df[self.features].values)
    
    def get_params(self):
        return self.model.get_params()


def compute_metrics_collection(actual, prediction):
    return  mean_squared_error(actual, prediction, squared=False)

