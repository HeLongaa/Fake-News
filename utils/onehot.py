# -*- coding: utf-8 -*-

import pickle

import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import config


def save_vocab(vocab, vocab_path):
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for k, v in vocab.items():
            f.write("%s\t%s\n" % (k, v))


class OnehotFeatureGenerator(object):
    def __init__(self, name='onehotFeatureGenerator'):
        self.name = name

    def process(self, df):
        print('process:', self.name)
        # 1). create strings based on text
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

        tokenizer = Tokenizer()
        data_set = df["text_unigram_str"].tolist()
        tokenizer.fit_on_texts(data_set)
        sequences = tokenizer.texts_to_sequences(data_set)

        vocabulary = tokenizer.word_index
        save_vocab(vocabulary, config.vocab_path)
        print('vocabulary size:%d' % len(vocabulary))
        print('vocabulary list:')
        count = 0
        for k, v in vocabulary.items():
            if count < 10:
                print("%s\t%s" % (k, v))
                count += 1

        # save train and test into separate files
        data_feature = pad_sequences(sequences, maxlen=300)
        print('Shape of Data Tensor:', data_feature.shape)
        train_labels = train['label'].values
        train_labels = to_categorical(train_labels, num_classes=len(pd.value_counts(train_labels)))
        print('Shape of Label Tensor:', train_labels.shape)

        onehot_train = data_feature[:n_train, :]
        onehot_train_feature_path = config.output_dir + "train.text.onehot.pkl"
        with open(onehot_train_feature_path, "wb") as f:
            pickle.dump(onehot_train, f)
        print('text onehot features of training set saved in %s' % onehot_train_feature_path)

        onehot_test = None
        if n_test > 0:
            # test set is available
            onehot_test = data_feature[n_train:, :]
            onehot_test_feature_path = config.output_dir + "test.text.onehot.pkl"
            with open(onehot_test_feature_path, "wb") as f:
                pickle.dump(onehot_test, f)
            print('text onehot features of test set saved in %s' % onehot_test_feature_path)
        return onehot_train, onehot_test, train_labels, vocabulary

    def read(self, header='train'):
        text_feature_path = config.output_dir + "%s.text.onehot.pkl" % header
        with open(text_feature_path, "rb") as f:
            text_onehot = pickle.load(f)

        print('text_onehot.shape:', text_onehot.shape)
        return text_onehot


def read_onehot_feature_label():
    data = pd.read_pickle(config.ngram_feature_path)
    return OnehotFeatureGenerator().process(data)


if __name__ == '__main__':
    data = pd.read_pickle(config.ngram_feature_path)

    OnehotFeatureGenerator().process(data)
