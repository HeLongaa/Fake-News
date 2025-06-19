# -*- coding: utf-8 -*-


from sklearn.linear_model import LogisticRegression

from models.base_model import BaseClassicModel


class LRModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='lr'):
        super().__init__(num_folds, name=name)

    def create_model(self):
        model = LogisticRegression()
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        model.fit(x_train, y_train)
