# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from features import ngram
from features.char_tfidf_feature import CharTfidfFeatureGenerator
from features.count_feature import CountFeatureGenerator
from features.onehot_feature import OnehotFeatureGenerator
from features.sentiment_feature import SentimentFeatureGenerator
from features.tfidf_feature import TfidfFeatureGenerator
from features.tokenizer import tokenizer
from features.word2vec_feature import Word2VecFeatureGenerator

import config
from features.svd_feature import SvdFeatureGenerator


def generate_features_label():
    # -----------------------load data--------------------
    if not os.path.exists(config.ngram_feature_path):
        data = pd.read_pickle(config.data_file_path)

        # debug
        if config.is_debug:
            data_a = data[:500]
            data_b = data[-100:]
            data = pd.concat((data_a, data_b))

        print('data shape:', data.shape)

        print("generate unigram")
        data["text_unigram"] = data["text"].map(lambda x: tokenizer(x))

        print("generate bigram")
        join_str = "_"
        data["text_bigram"] = data["text_unigram"].map(lambda x: ngram.getBigram(x, join_str))

        print("generate trigram")
        data["text_trigram"] = data["text_unigram"].map(lambda x: ngram.getTrigram(x, join_str))

        data["text_unigram_str"] = data["text_unigram"].map(lambda x: ' '.join(x))
        print(data.head())

        data.to_pickle(config.ngram_feature_path)
        print('data ngram features saved in ', config.ngram_feature_path)
    else:
        data = pd.read_pickle(config.ngram_feature_path)
        # debug
        if config.is_debug:
            data_a = data[:500]
            data_b = data[-100:]
            data = pd.concat((data_a, data_b))

    # feature generators
    generators = [CountFeatureGenerator(),
                  CharTfidfFeatureGenerator(),
                  TfidfFeatureGenerator(),
                  SvdFeatureGenerator(),
                  Word2VecFeatureGenerator(),
                  SentimentFeatureGenerator(),
                  OnehotFeatureGenerator()]
    for g in generators:
        g.process(data)

    print('done')


def read_features_label():
    data = pd.read_pickle(config.ngram_feature_path)
    # debug
    if config.is_debug:
        data_a = data[:500]
        data_b = data[-100:]
        data = pd.concat((data_a, data_b))

    generators = [CountFeatureGenerator(),
                  SvdFeatureGenerator(),
                  Word2VecFeatureGenerator(),
                  SentimentFeatureGenerator()]

    # build data
    print("read feature labels data...")
    train_features = [f for g in generators for f in g.read('train')]
    train_features = [f.toarray() if isinstance(f, csr_matrix) else f for f in train_features]
    for i, f in enumerate(train_features):
        print('shape: ', i, f.shape)
    train_data_x = np.hstack(train_features)
    print('train data_x.shape:', train_data_x.shape)

    test_features = [f for g in generators for f in g.read('test')]
    test_features = [f.toarray() if isinstance(f, csr_matrix) else f for f in test_features]
    test_data_x = np.hstack(test_features)
    print('test data_x.shape:', test_data_x.shape)

    data_y = data['label'].values
    return train_data_x, test_data_x, data_y


if __name__ == "__main__":
    generate_features_label()
