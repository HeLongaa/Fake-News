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
import joblib
from tqdm import tqdm
from xgboost import callback as xgb_callback
from catboost import Pool

from models.base_model import BaseClassicModel
import config

'''
逻辑回归模型
'''
class LRModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='lr', model_path=None):
        super().__init__(num_folds, name=name)
        self.model_path = model_path or (config.models_dir + 'lr.model')

    def create_model(self):
        model = LogisticRegression()
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        model.fit(x_train, y_train)

'''
XGBoost模型
'''
class XgboostModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='xgboost', model_path=None):
        super().__init__(num_folds, name=name)
        self.model_path = model_path or (config.models_dir + 'xgboost.model')

    def create_model(self):
        model = XGBClassifier(max_depth=3,
                              learning_rate=0.1,
                              n_estimators=300,
                              eval_metric='logloss')
        return model

    def fit_model(
        self, model, x_train, y_train, x_valid, y_valid):
        model.fit(
            x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=10
        )

'''
CatBoost模型
'''
class CatBoostModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='catboost', model_path=None):
        super().__init__(num_folds, name=name)
        self.model_path = model_path or (config.models_dir + 'catboost.model')

    def create_model(self):
        model = CatBoostClassifier(loss_function='Logloss', depth=8, n_estimators=500, logging_level='Silent')
        return model

    def fit_model(
        self, model, x_train, y_train, x_valid, y_valid):
        train_pool = Pool(x_train, y_train)
        eval_pool = Pool(x_valid, y_valid)
        n_iters = model.get_params().get('n_estimators', 100)
        for i in tqdm(range(1, n_iters + 1), desc="CatBoost Training", ncols=80):
            model.fit(
                train_pool,
                eval_set=eval_pool,
                init_model=model if i > 1 else None,
                iterations=1,
                use_best_model=False,
                verbose=False
            )
