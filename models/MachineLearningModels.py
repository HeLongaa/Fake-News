# -*- coding: utf-8 -*-
"""
@description: 传统机器学习模型
1. LR Model
2. XGBoost Model
3. CatBoost Model
"""
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from models.base_model import BaseClassicModel

'''
逻辑回归模型
'''
class LRModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='lr'):
        super().__init__(num_folds, name=name)

    def create_model(self):
        model = LogisticRegression()
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        model.fit(x_train, y_train)

'''
XGBoost模型
'''
class XgboostModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='xgboost'):
        super().__init__(num_folds, name=name)

    def create_model(self):
        model = XGBClassifier(max_depth=3,
                              learning_rate=0.1,
                              n_estimators=300)
        return model

    def fit_model(
        self, model, x_train, y_train, x_valid, y_valid):
        model.fit(
            x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=3)

'''
CatBoost模型
'''
class CatBoostModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='catboost'):
        super().__init__(num_folds, name=name)

    def create_model(self):
        model = CatBoostClassifier(loss_function='Logloss', depth=8, n_estimators=500)
        return model

    def fit_model(
        self, model, x_train, y_train, x_valid, y_valid):
        model.fit(
            x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=3)
