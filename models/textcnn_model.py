# -*- coding: utf-8 -*-


from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D
from keras.layers import Embedding
from keras.layers.merge import Concatenate
from keras.models import Model

import config
from models.base_model import BaseDeepModel


class TextCNNModel(BaseDeepModel):
    def __init__(self, max_len=300,
                 num_folds=1,
                 name='textcnn',
                 filter_sizes='4,5,6',
                 embedding_dim=128,
                 hidden_dim=128,
                 num_filters=512,
                 num_classes=2,
                 batch_size=64,
                 vocabulary_size=20000,
                 dropout=0.5,
                 num_epochs=1,
                 model_path=config.output_dir + 'textcnn.model'):
        if "," in filter_sizes:
            self.filter_sizes = filter_sizes.split(",")
        else:
            self.filter_sizes = [3, 4, 5]
        self.dropout = dropout
        self.num_filters = num_filters
        self.model_path = model_path
        super(TextCNNModel, self).__init__(max_len=max_len,
                                           num_folds=num_folds,
                                           name=name,
                                           num_classes=num_classes,
                                           vocabulary_size=vocabulary_size,
                                           embedding_dim=embedding_dim,
                                           hidden_dim=hidden_dim,
                                           batch_size=batch_size,
                                           num_epochs=num_epochs)

    def create_model(self):
        print("Creating text CNN Model...")
        # a tensor
        inputs = Input(shape=(self.max_len,), dtype='int32')
        # emb
        embedding = Embedding(input_dim=self.vocabulary_size,
                              output_dim=self.embedding_dim,
                              input_length=self.max_len,
                              name="embedding")(inputs)
        # convolution block
        conv_blocks = []
        for sz in self.filter_sizes:
            conv = Convolution1D(filters=self.num_filters,
                                 kernel_size=int(sz),
                                 strides=1,
                                 padding='valid',
                                 activation='relu')(embedding)
            conv = MaxPooling1D()(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        conv_concate = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        dropout_layer = Dropout(rate=self.dropout)(conv_concate)
        output = Dense(self.hidden_dim, activation='relu')(dropout_layer)
        output = Dense(self.num_classes, activation='softmax')(output)
        # model
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        cp = ModelCheckpoint(self.model_path, monitor='val_acc', verbose=1, save_best_only=True)
        # fit and save model
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                  validation_data=(x_valid, y_valid), callbacks=[cp, es])
