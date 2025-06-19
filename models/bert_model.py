# -*- coding: utf-8 -*-

import os
import sys

import kashgari
import pandas as pd
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import KMax_CNN_Model

sys.path.append('..')
import config
from models.base_model import BaseDeepModel
from models.bert_tokenization import BasicTokenizer


def read_bert_feature_label():
    tokenizer = BasicTokenizer()

    df = pd.read_pickle(config.ngram_feature_path)
    train = df[df['type'] == 'train']
    print(train.head())

    test = df[df['type'] == 'test']
    print(test.head())
    print('train.shape:', train.shape)
    print('test.shape:', test.shape)

    n_train = train.shape[0]
    print('n_train:', n_train)
    n_test = test.shape[0]
    print('n_test:', n_test)

    df['bert_tokens'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
    train_x = df['bert_tokens'][:n_train].tolist()
    test_x = df['bert_tokens'][n_train:].tolist()
    train_y = df['label'][:n_train].tolist()

    print('len train_x:', len(train_x))
    print('len test_x:', len(test_x))
    print('len train_y:', len(train_y))
    return train_x, test_x, train_y


class BertModel(BaseDeepModel):
    def __init__(self, bert_path=config.pretrained_bert_path,
                 max_len=200,
                 name='bert',
                 num_classes=2,
                 batch_size=32,
                 num_epochs=10,
                 model_path=config.output_dir + 'bert_kmaxcnn_model'):
        self.bert_path = bert_path
        self.model_path = model_path
        super(BertModel, self).__init__(max_len=max_len,
                                        name=name,
                                        num_classes=num_classes,
                                        batch_size=batch_size,
                                        num_epochs=num_epochs)

    def create_model(self):
        if os.path.exists(self.model_path):
            print("Loading bert embedding KMax CNN Model...")
            model = kashgari.utils.load_model(self.model_path)
        else:
            print("Creating bert embedding KMax CNN Model...")

            # 初始化 Embedding
            embed = BERTEmbedding(model_folder=self.bert_path,
                                  task=kashgari.CLASSIFICATION,
                                  sequence_length=self.max_len)

            # 使用 embedding 初始化模型
            model = KMax_CNN_Model(embed)
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        model.fit(x_train, y_train, x_valid, y_valid,
                  batch_size=self.batch_size, epochs=self.num_epochs)
        if not config.is_debug:
            model.save(self.model_path)


if __name__ == '__main__':
    train_x, test_x, train_y = read_bert_feature_label()
    m = BertModel(name='bert',
                  num_classes=2,
                  batch_size=32,
                  num_epochs=1)
    # score = m.train_predict(train_x, train_y, test_x, predict_path)
    b_model = kashgari.utils.load_model(config.output_dir + 'bert.model')
    y_pred_train = b_model.predict(train_x[:5])
    from models.score import compute_acc

    train_y = [str(i) for i in train_y[:5]]
    accuracy_rate = compute_acc(train_y, y_pred_train)
    print(m.name, accuracy_rate)
