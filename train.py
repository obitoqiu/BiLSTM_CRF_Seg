
# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint
from LSTM_model import *
from process import *
import numpy as np
import os


def random_split(x, y, split=0.2):
    indices = np.arange(x.shape[0])
    x = x[indices]
    y = y[indices]
    num_train_samples = int(split * x.shape[0])
    x_train = x
    y_train = y
    x_val, y_val, word_val_index, chunk_val_index = process_data('分词\\test.bis.txt')

    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    DATA_PATH = '分词\\bis.txt'
    #DATA_PATH = 'E:\\pycharm\\LSTM_Seg\\分词\\bis.txt'
    VAL_SPLIT = 0.3

    SAVE_DIR = ''
    #SAVE_DIR = 'E:\\pycharm\\LSTM_Seg'

    # EMBEDDING_FILE_PATH = 'D:\\embidding\\BAIDU_300_embedding.iter5'
    EMBEDDING_FILE_PATH ='tmp\\mymodel_50.txt'
    #EMBEDDING_FILE_PATH = 'E:\\pycharm\\LSTM_Seg\\tmp\\mymodel.txt'
    MODEL_DIR = None
    EPOCHS = 10

    x, y, word_index, chunk_index = process_data(DATA_PATH)

    x_train, y_train, x_val, y_val = random_split(x, y, VAL_SPLIT)

    if MODEL_DIR is not None:
        word_index, chunk_index = load_dict(os.path.join(MODEL_DIR, 'model.dict'))
        model_configure = load_model_config(os.path.join(MODEL_DIR, 'model.cfg'))
        model = model_configure.build_model()
        model.load_weights(os.path.join(MODEL_DIR, 'model.final.h5'))

    else:
        model_configure = LSTMModelConfigure(len(word_index) + 1, len(chunk_index) + 1)
        model_configure.save_model_config(model_configure, os.path.join(SAVE_DIR, 'model.cfg'))
        model_configure.save_dict((word_index, chunk_index), os.path.join(SAVE_DIR, 'model.dict'))
        embedding_matrix = None
        if EMBEDDING_FILE_PATH is not None:
            embedding_matrix = create_embedding_matrix(get_embedding_index(EMBEDDING_FILE_PATH), word_index,
                                                       model_configure)

        model = model_configure.build_model(embedding_matrix)


    model.summary()
    check = ModelCheckpoint(os.path.join(SAVE_DIR, 'weights.{epoch:02d}-{val_loss:.2f}.h5'),
                            monitor='loss', verbose=0)

    model.fit(x_train, y_train, batch_size=64, epochs=EPOCHS,
              validation_data=(x_val, y_val), callbacks=[check])
    model.save(os.path.join(SAVE_DIR, 'model.final.h5'))