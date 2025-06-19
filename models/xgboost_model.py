# -*- coding: utf-8 -*-


from xgboost import XGBClassifier

from models.base_model import BaseClassicModel


class XgboostModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='xgboost'):
        super().__init__(num_folds, name=name)

    def create_model(self):
        model = XGBClassifier(max_depth=3,
                              learning_rate=0.1,
                              n_estimators=300)
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=3)
