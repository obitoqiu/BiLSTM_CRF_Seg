
# -*- coding: utf-8 -*-
from LSTM_model import *
from process import*
import pickle
import numpy as np
from keras.utils import to_categorical


def predict(sentences, word_index, index_chunk, model, model_config):
    x = sentence_to_vec(sentences, word_index, model_config)
    preds = model.predict(x, batch_size=1024)
    order = 0
    sen = ''
    index = 0
    bis =[]
    bises = []

    for pred in preds:
        tags = []
        for word in pred:
            tag_index = np.argmax(word)
            if index_chunk.get(tag_index) is not None:
                tags.append(index_chunk.get(tag_index))

        bis = bis+to_bis(sentences[order], tags)
        if sentences[order][-1]=='\n':
            bises.append(bis)
            bis = []
        order = order + 1

    save_bis(bises,'分词\\pre.bis.txt' )
    bis_to_text(bises,'分词\\test-pre.txt')


def to_bis(sentence,tags):
    bis = []
    words = list(sentence)
    for i, tag in enumerate(tags):
        if tag == 'b':
           bis.append((words[i],tag.upper()))
        elif tag == 'i':
            bis.append((words[i], tag.upper()))
        elif tag == 's':
            if words[i] == '\n':
                continue
            else:
                bis.append((words[i], tag.upper()))
    return bis


def save_bis(bises, pre_bis_path):
    with open(pre_bis_path, 'w', encoding='UTF-8')as f:
        for bis in bises:
            for char, tag in bis:
                f.write(char + ' ' + tag + '\n')
            f.write('\n')


def bis_to_text(bises,  pref_file_patt):
    text = ''
    for bis in bises:
        cuts, t1 =[], []
        for char, tag in bis:
            if tag == 'B':
                if len(t1) != 0:
                    cuts.append(t1)
                    t1=[]
                t1.append(char)
            elif tag == 'I':
                t1.append(char)
            elif tag == 'S':
                if len(t1) != 0:
                    cuts.append(t1)
                    t1=[]
                t1.append(char)
                cuts.append(t1)
                t1 = []
            if (char,tag) == bis[-1] and len(t1) != 0:
                cuts.append(t1)

        for cut in cuts:
            str=''
            for w in cut:
                temp=''.join(w)
                str = str + temp
            text = text + str +'  '
        text = text + '\n'
    with open(pref_file_path, 'w', encoding='UTF-8')as f:
        f.write(text)


def cut_sentence(sentence, tags):
    words = list(sentence)
    cuts, t1 = [], []
    for i, tag in enumerate(tags):
        if tag == 'b':
            if len(t1) != 0:
                cuts.append(t1)
            t1 = [words[i]]
            # print(words[i]+' '+tag)
        elif tag == 'i':
            t1.append(words[i])
            # print(words[i] + ' ' + tag)
        elif tag == 's':
            if len(t1) != 0:
                cuts.append(t1)
            cuts.append([words[i]])
            # print(words[i] + ' ' + tag)
            t1 = []
        if i == len(tags) - 1 and len(t1) != 0:
            cuts.append(t1)
    return cuts


def cuts_to_str(cuts):
    str = ''.join(cuts[0])
    for t in cuts[1:]:
        temp = '  '+''.join(t)
        str = str+temp
    return str


def _load_sentences(text_file_path):
    small_sentence = []
    splits = ['，', '、', '；']
    with open(text_file_path, 'r', encoding='UTF-8') as f:
        sentences = f.readlines()
        for sentence in sentences:
            sentence = list(sentence)
            t = []
            for w in sentence:
                if w in splits and len(t) != 0:
                    small_sentence.append(t)
                    t = []
                    t.append(w)
                elif w == '\n':
                    t.append(w)
                    small_sentence.append(t)
                    t = []
                else:
                    t.append(w)
    f.close()
    return small_sentence


def _save_pred(text, pred_file_path):
    with open(pred_file_path,'w',encoding='UTF-8')as f:
        f.write(text)


if __name__ == '__main__':
    model_base_dir = ''
    #model_base_dir = 'E:\\pycharm\\LSTM_Seg\\'

    text_file_path = '分词\\test.txt'
    #text_file_path = 'E:\\pycharm\\LSTM_Seg\\分词\\test.txt'

    pref_file_path = '分词\\test-pre.txt'
    #pref_file_path= 'E:\\pycharm\\LSTM_Seg\\分词\\test-pre.txt'

    pre_bis_path = '分词\\pre.bis.txt'
    test_bis_path = '分词\\test.bis.txt'

    with open(os.path.join(model_base_dir, 'model.cfg'), 'rb') as f:
        config = pickle.load(f)
    with open(os.path.join(model_base_dir, 'model.dict'), 'rb') as f:
        word_index, chunk_index = pickle.load(f)

    model = config.build_model()
    model.load_weights('weights.10--0.42.h5')  # 'model.final.h5'
    index_chunk = {i: c for c, i in chunk_index.items()}
    print(word_index)
    '''
    sentences = _load_sentences(text_file_path)
    # print(sentences)

    result = predict(sentences, word_index, index_chunk, model, config)
    '''

