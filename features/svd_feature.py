# -*- coding: utf-8 -*-

import pickle

from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD

import config
from features.tfidf_feature import TfidfFeatureGenerator


class SvdFeatureGenerator(object):
    def __init__(self, name='svdFeatureGenerator'):
        self.name = name

    def process(self, df):
        train = df[df['type'] == 'train']
        print(train.head())

        test = df[df['type'] == 'test']
        print(test.head())
        print('train.shape:', train.shape)

        n_train = train.shape[0]
        print('n_train:', n_train)
        n_test = test.shape[0]
        print('n_test:', n_test)

        tfidf_generator = TfidfFeatureGenerator()
        features_train = tfidf_generator.read('train')
        text_tfidf_train = features_train[0]

        text_tfidf = text_tfidf_train
        if n_test > 0:
            # test set is available
            features_test = tfidf_generator.read('test')
            text_tfidf_test = features_test[0]
            text_tfidf = vstack([text_tfidf_train, text_tfidf_test])

        # compute the cosine similarity between truncated-svd features
        svd = TruncatedSVD(n_components=250, n_iter=15)
        svd.fit(text_tfidf)  # fit to the combined train-test set (or the full training set for cv process)
        print('text Tfidf.shape:', text_tfidf.shape)
        text_svd = svd.transform(text_tfidf)
        print('text svd.shape:', text_svd.shape)

        text_svd_train = text_svd[:n_train, :]
        text_svd_train_path = config.output_dir + "train.text.svd.pkl"
        with open(text_svd_train_path, "wb") as f:
            pickle.dump(text_svd_train, f)
        print('text svd features of training set saved in %s' % text_svd_train_path)

        if n_test > 0:
            # test set is available
            text_svd_test = text_svd[n_train:, :]
            text_svd_test_path = config.output_dir + "test.text.svd.pkl"
            with open(text_svd_test_path, "wb") as f:
                pickle.dump(text_svd_test, f)
            print('text svd features of test set saved in %s' % text_svd_test_path)

        # sim_svd = np.asarray(map(cosine_sim, text_svd, text_svd))[:, np.newaxis]
        # print('sim svd.shape:', sim_svd.shape)
        #
        # sim_svd_train = sim_svd[:n_train]
        # sim_svd_train_path = config.output_dir + "train.sim.svd.pkl"
        # with open(sim_svd_train_path, "wb") as f:
        #     pickle.dump(sim_svd_train, f)
        # print('sim svd features of training set saved in %s' % sim_svd_train_path)
        #
        # if n_test > 0:
        #     # test set is available
        #     sim_svd_test = sim_svd[n_train:]
        #     sim_svd_test_path = config.output_dir + "test.sim.svd.pkl"
        #     with open(sim_svd_test_path, "wb") as f:
        #         pickle.dump(sim_svd_test, f)
        #     print('sim svd features of test set saved in %s' % sim_svd_test_path)

    def read(self, header='train'):
        text_svd_feature_path = config.output_dir + "%s.text.svd.pkl" % header
        with open(text_svd_feature_path, "rb") as f:
            text_svd = pickle.load(f)

        # body_svd_feature_path = config.output_dir +"%s.body.svd.pkl" % header
        # with open(body_svd_feature_path, "rb") as f:
        #     body_svd = pickle.load(f)

        # sim_svd_feature_path = config.output_dir + "%s.sim.svd.pkl" % header
        # with open(sim_svd_feature_path, "rb") as f:
        #     sim_svd = pickle.load(f)

        print('text_svd.shape:', text_svd.shape)
        # print('body_svd.shape:', body_svd.shape)
        # print('sim_svd.shape:', sim_svd.shape)
        # return [text_svd, sim_svd.reshape(-1, 1)]
        return [text_svd]
