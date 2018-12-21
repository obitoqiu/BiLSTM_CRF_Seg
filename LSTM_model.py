
# -*- coding: utf-8 -*-
from process import *
from keras.layers import *
from keras.models import *
from keras_contrib.layers import CRF
import keras
import pickle


class LSTMModelConfigure:
    def __init__(self, vocab_size
                 , chunk_size
                 , embed_dim=50
                 , lstm_units=128
                 , max_sequence_len=50
                 , max_num_words=20000
                 ):
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        self.bi_lstm_units = lstm_units
        self.max_sequence_len = max_sequence_len
        self.max_num_words = max_num_words
        self.num_words = min(self.max_num_words, self.vocab_size)

    def build_model(self, embeddings_matrix=None):

        word_input = Input(shape=(self.max_sequence_len,), dtype='int32', name='word_input')
        if embeddings_matrix is not None:
            word_embedding = Embedding(self.num_words, self.embed_dim
                                       , input_length=self.max_sequence_len
                                       , weights=[embeddings_matrix]
                                       , trainable=True
                                       , name='word_emb')(word_input)

        else:
            word_embedding = Embedding(self.num_words, self.embed_dim
                                       , input_length=self.max_sequence_len
                                       , name='word_emb')(word_input)
        # bilstm + second_bilstm
        bilstm = Bidirectional(LSTM(self.bi_lstm_units // 2,  return_sequences=True))\
            (word_embedding)
        sencnd_bilstm = Bidirectional(LSTM(self.bi_lstm_units // 2,  return_sequences=True))\
            (bilstm)

        x = Dropout(0.4)(sencnd_bilstm)
        dense = TimeDistributed(Dense(self.chunk_size, activation='softmax'))(x)

        # crf
        crf = CRF(self.chunk_size, sparse_target=False)
        crf_output = crf(dense)

        model = Model([word_input], [crf_output])
        model.compile(optimizer=keras.optimizers.Adam(), loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    def save_dict(self, dict, dict_path):
        with open(dict_path, 'wb') as f:
            pickle.dump(dict, f)

    def save_model_config(self, model_config, model_config_path):
        with open(model_config_path, 'wb') as f:
            pickle.dump(model_config, f)

    def load_model_config(self, model_config_path):
        with open(model_config_path, 'rb') as f:
            model_builder = pickle.load(f)
        return model_builder

    def load_dict(self, dict_path):
        with open(dict_path, 'rb') as f:
            vocab, chunk = pickle.load(f)
        return vocab, chunk
