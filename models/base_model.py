# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from models.score import compute_acc, write_list


class BaseDeepModel(object):
    """
    basic class of deep models
    """

    def __init__(self, vocabulary_size=20000,
                 max_len=300,
                 name='base_deep_model',
                 num_folds=1,
                 batch_size=64,
                 num_classes=2,
                 embedding_dim=128,
                 hidden_dim=128,
                 num_epochs=10):
        self.name = name
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocabulary_size = vocabulary_size + 1
        self.num_folds = 1 if num_folds <= 1 else num_folds
        self.num_epochs = num_epochs
        self.model = None

    def create_model(self):
        raise NotImplementedError("need impl create model.")

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError("need impl fit model.")

    def train_predict(self, train_x, train_y, test_x, predict_path):
        """
        train model and predict test result
        :param train_x: list or array
        :param train_y: list or array
        :param test_x: list or array
        :param predict_path: str
        :return:
        """
        if not self.model:
            self.model = self.create_model()
        print('train and predict with model:', self.name)
        if not isinstance(train_x, list):
            print('train_x.shape:', train_x.shape)
            print('train_y.shape:', train_y.shape)
            print('test_x.shape:', test_x.shape)

        n_train = len(train_x)
        n_test = len(test_x)
        train_y = train_y[:n_train]
        print('n_train:', n_train)
        print('n_test:', n_test)
        print('n_train_y:', len(train_y))
        print('num_classes:', self.num_classes)

        stack = np.zeros((n_train, self.num_classes))
        stack_test = np.zeros((n_test, self.num_classes))
        scores = []
        if self.num_folds > 1:
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=10)
            for i, (tr, va) in enumerate(kf.split(train_y)):
                print('%s stack:%d/%d' % (str(datetime.now()), i + 1, self.num_folds))
                x_train = train_x[tr]
                y_train = train_y[tr]
                x_valid = train_x[va]
                y_valid = train_y[va]
                self.fit_model(self.model, x_train, y_train, x_valid, y_valid)
                y_pred_valid = self.model.predict(x_valid)
                y_pred_test = self.model.predict(test_x)
                accuracy_rate = compute_acc(y_valid, y_pred_valid)
                print('valid acc:', accuracy_rate)
                scores.append(accuracy_rate)
                stack[va] += y_pred_valid
                stack_test += y_pred_test
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.1)
            self.fit_model(self.model, x_train, y_train, x_valid, y_valid)
            y_pred_valid = self.model.predict(x_valid)
            y_pred_test = self.model.predict(test_x)
            accuracy_rate = compute_acc(y_valid, y_pred_valid)
            if isinstance(y_pred_test, list):
                # 兼容bert模型的输出类型
                write_list(y_pred_test, predict_path)
                return accuracy_rate
            print('valid acc:', accuracy_rate)
            scores.append(accuracy_rate)
            y_pred_train = self.model.predict(train_x)
            stack += y_pred_train
            stack_test += y_pred_test
        stack_test /= self.num_folds
        score = np.mean(scores)
        print("total scores:", scores, " mean score:", score)
        # output predict result
        data = pd.DataFrame()
        stack_all = np.vstack([stack, stack_test])
        for i in range(stack_all.shape[1]):
            data['{}_{}'.format('label', i)] = stack_all[:, i]
        data['predict_label'] = np.argmax(stack_all, axis=1)
        data.to_csv(predict_path, encoding='utf-8')
        print(datetime.now(), ' predict result save to', predict_path)
        return score


class BaseClassicModel(object):
    """
    basic class of classic models
    """

    def __init__(self, num_folds=0, name='base_static_model'):
        self.name = name
        self.num_folds = 1 if num_folds <= 1 else num_folds
        self.model = None

    def create_model(self):
        raise NotImplementedError("need impl create model.")

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError("need impl fit model.")

    def train_predict(self, train_x, train_y, test_x, predict_path):
        """
        train model and predict test result
        :param train_x: list or array
        :param train_y: list or array
        :param test_x: list
        :param predict_path: str
        :return:
        """
        if not self.model:
            self.model = self.create_model()
        print('train and predict with model:', self.name)

        n_train = len(train_x)
        n_test = len(test_x)
        train_y = train_y[:n_train]
        print('n_train:', n_train)
        print('n_test:', n_test)
        print('n_train_y:', len(train_y))
        print('y value_counts:\n', pd.value_counts(train_y))
        num_classes = len(pd.value_counts(train_y))
        print('num_classes:', num_classes)

        stack = np.zeros((n_train, num_classes))
        stack_test = np.zeros((n_test, num_classes))
        scores = []
        if self.num_folds > 1:
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=10)
            for i, (tr, va) in enumerate(kf.split(train_y)):
                print('%s stack:%d/%d' % (str(datetime.now()), i + 1, self.num_folds))
                x_train = train_x[tr]
                y_train = train_y[tr]
                x_valid = train_x[va]
                y_valid = train_y[va]
                self.fit_model(self.model, x_train, y_train, x_valid, y_valid)
                y_pred_valid = self.model.predict_proba(x_valid)
                y_pred_test = self.model.predict_proba(test_x)
                accuracy_rate = compute_acc(y_valid, y_pred_valid)
                print('valid acc:', accuracy_rate)
                scores.append(accuracy_rate)
                stack[va] += y_pred_valid
                stack_test += y_pred_test
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.1)
            self.fit_model(self.model, x_train, y_train, x_valid, y_valid)
            y_pred_valid = self.model.predict_proba(x_valid)
            y_pred_test = self.model.predict_proba(test_x)
            y_pred_train = self.model.predict_proba(train_x)
            accuracy_rate = compute_acc(y_valid, y_pred_valid)
            print('valid acc:', accuracy_rate)
            scores.append(accuracy_rate)
            stack += y_pred_train
            stack_test += y_pred_test
        stack_test /= self.num_folds
        score = np.mean(scores)
        print("total scores:", scores, " mean score:", score)
        # output predict result
        data = pd.DataFrame()
        stack_all = np.vstack([stack, stack_test])
        for i in range(stack_all.shape[1]):
            data['{}_{}'.format('label', i)] = stack_all[:, i]
        data['predict_label'] = np.argmax(stack_all, axis=1)
        data.to_csv(predict_path, encoding='utf-8')
        print(datetime.now(), ' predict result save to', predict_path)
        return score
