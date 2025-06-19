# -*- coding: utf-8 -*-

import os
import pickle
from functools import reduce

import gensim
import numpy as np
from sklearn.preprocessing import normalize

import config


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


class Word2VecFeatureGenerator(object):
    def __init__(self, name='word2vecFeatureGenerator'):
        self.name = name

    def process(self, df):
        print('process:', self.name)
        print('generating word2vec features')
        train = df[df['type'] == 'train']
        print(train.head())

        test = df[df['type'] == 'test']
        print(test.head())
        print('train.shape:', train.shape)

        n_train = train.shape[0]
        print('n_train:', n_train)
        n_test = test.shape[0]
        print('n_test:', n_test)

        # 1). document vector built by multiplying together all the word vectors
        w2v_bin_path = config.output_dir + 'text_w2v.bin'
        if not os.path.exists(w2v_bin_path):
            # train pre-trained word vectors
            text_segment = df['text_unigram_str'].tolist()
            text_segment_path = config.output_dir + 'text_segment.txt'
            save_sentence(text_segment, text_segment_path)
            print('train w2v model...')
            # train model
            w2v = gensim.models.Word2Vec(sg=1, sentences=gensim.models.word2vec.LineSentence(text_segment_path),
                                         vector_size=300, window=5, min_count=5, epochs=40)
            w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
            print("save w2v model: %s." % w2v_bin_path)

        # load word vectors
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
        print('model loaded')

        text_unigram_array = df['text_unigram'].values
        print('text_unigram_array:', text_unigram_array.tolist()[:5])
        print('text_unigram_array.shape:', text_unigram_array.shape)

        # word vectors weighted by normalized tf-idf coefficient?
        text_vec = list(map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.] * 300),
                            text_unigram_array))
        text_vec = np.array(text_vec)
        print("text_vec:", text_vec)
        text_vec = normalize(text_vec)
        print("normalized text_vec:", text_vec)
        print(text_vec.shape)

        text_vec_train = text_vec[:n_train, :]
        text_vec_train_path = config.output_dir + "train.text.word2vec.pkl"
        with open(text_vec_train_path, "wb") as f:
            pickle.dump(text_vec_train, f)
        print('text word2vec features of training set saved in %s' % text_vec_train_path)

        if n_test > 0:
            # test set is available
            text_vec_test = text_vec[n_train:, :]
            text_vec_test_path = config.output_dir + "test.text.word2vec.pkl"
            with open(text_vec_test_path, "wb") as f:
                pickle.dump(text_vec_test, f)
            print('text word2vec features of test set saved in %s' % text_vec_test_path)
        print('w2v text done')

        # compute cosine similarity between text/body word2vec features
        # sim_vec = np.asarray(map(cosine_sim, text_vec, text_vec))[:, np.newaxis]
        # print('sim vec.shape:', sim_vec.shape)
        #
        # sim_vec_train = sim_vec[:n_train]
        # sim_vec_train_path = config.output_dir + "train.sim.word2vec.pkl"
        # with open(sim_vec_train_path, "wb") as f:
        #     pickle.dump(sim_vec_train, f)
        # print('word2vec sim. features of training set saved in %s' % sim_vec_train_path)
        #
        # if n_test > 0:
        #     # test set is available
        #     sim_vec_test = sim_vec[n_train:]
        #     sim_vec_test_path = config.output_dir + "test.sim.word2vec.pkl"
        #     with open(sim_vec_test_path, "wb") as f:
        #         pickle.dump(sim_vec_test, f)
        #     print('word2vec sim. features of test set saved in %s' % sim_vec_test_path)

    def read(self, header='train'):
        text_vec_path = config.output_dir + "%s.text.word2vec.pkl" % header
        with open(text_vec_path, "rb") as f:
            text_vec = pickle.load(f)

        # body_vec_path =config.output_dir +  "%s.body.word2vec.pkl" % header
        # with open(body_vec_path, "rb") as f:
        #     body_vec = pickle.load(f)

        # sim_vec_path = config.output_dir + "%s.sim.word2vec.pkl" % header
        # with open(sim_vec_path, "rb") as f:
        #     sim_vec = pickle.load(f)

        # return [text_vec, sim_vec]
        return [text_vec]
