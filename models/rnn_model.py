# -*- coding: utf-8 -*-


from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model

import config
from models.base_model import BaseDeepModel


class RNNModel(BaseDeepModel):
    def __init__(self, max_len=300,
                 num_folds=1,
                 name='rnn',
                 embedding_dim=300,
                 hidden_dim=128,
                 num_classes=2,
                 batch_size=64,
                 vocabulary_size=20000,
                 num_epochs=1,
                 model_path=config.output_dir + 'rnn.model'):
        self.model_path = model_path
        super(RNNModel, self).__init__(max_len=max_len,
                                       num_folds=num_folds,
                                       name=name,
                                       num_classes=num_classes,
                                       vocabulary_size=vocabulary_size,
                                       embedding_dim=embedding_dim,
                                       hidden_dim=hidden_dim,
                                       batch_size=batch_size,
                                       num_epochs=num_epochs)

    def create_model(self):
        print("Creating bi-lstm Model...")
        # a tensor
        inputs = Input(shape=(self.max_len,), dtype='int32')
        # emb
        embedding = Embedding(input_dim=self.vocabulary_size,
                              output_dim=self.embedding_dim,
                              input_length=self.max_len,
                              name="embedding")(inputs)
        lstm_layer = Bidirectional(LSTM(self.hidden_dim))(embedding)
        output = Dense(self.num_classes, activation='softmax')(lstm_layer)
        model = Model(inputs, output)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.summary()
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        cp = ModelCheckpoint(self.model_path, monitor='val_acc', verbose=1, save_best_only=True)
        # fit and save model
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                  validation_data=(x_valid, y_valid), callbacks=[cp, es])
