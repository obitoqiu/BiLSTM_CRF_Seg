
# -*- coding: utf-8 -*-
from keras.preprocessing.text import Tokenizer
from gensim.models.word2vec import Word2Vec
import warnings
import re
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')


def max_seq(path):
    with open(path,'r',encoding ='UTF-8')as f:
        lines = f.readlines()
        max = 0
        id=0
        max_id=0
        for line in lines:
            str=''
            for word in line:
                temp=''.join(word)
                str=str+temp

            if id ==55960:
                print(str)
            str = re.split('／|，|；|  ',str)
            for short in str:
                short = short.replace('  ','')
                if len(short)>max:
                    max = len(short)
                    max_id = id
            id = id + 1
    print(max_id)

    return max


def _load_sentences(text_file_path):
    small_sentence = []
    splits = ['，', '；']
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


def w2v(corpus_file_path):
    text = []
    with open(corpus_file_path, 'r', encoding='UTF-8')as f:
        sentences = f.readlines()
        for sentence in sentences:
            sentence = sentence.replace('  ', '')
            text.append(list(sentence))

    model = Word2Vec(sentences, size=50, min_count=1)
    # model.build_vocab()
    model.save('tmp\\mymodel')
    model.wv.save_word2vec_format('tmp\\mymodel_50.txt', binary=False)
    # print(model['国'])


if __name__ == '__main__':
    model_base_dir = ''
    text_file_path = '分词\\test.txt'
    corpus_file_path = '分词\\train.txt'
    embedding_file_path = ''
    w2v(corpus_file_path)
    #print(max_seq(corpus_file_path))













