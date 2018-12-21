
# -*- coding: utf-8 -*-
import re


def convert2bis(source_path, target_path):
    bises = process_file(source_path)
    _save_bises(bises, target_path)


def _save_bises(bises, target_path):
    with open(target_path, 'w', encoding='UTF-8')as f:
        for bis in bises:
            for char,tag in bis:
                f.write(char+' '+tag+'\n')
            f.write('\n')


def process_file(file):
    with open(file, 'r', encoding='UTF-8')as f:
        text = f.readlines()
        bises = _parse_text(text)
    return bises


def _parse_text(text : list):
    bises = []
    for line in text:
        if line == '\n':
            continue
        # print(line)
        bises.append(_tag(line))

    return bises


# 给指定一行文本打上BIS标签
def _tag(line):
    bis = []
    words = re.split('\s', line)
    words = list(map(list, words))

    pre_word = None
    for word in words:
        if len(word)==0:
            continue
        if word[0]=='[':
            pre_word = word
            continue
        if pre_word is not None:
            pre_word += word
            if word[-1]!=']':
                continue
            else:
                word = pre_word[1:-1]
                pre_word = None

        if len(word)==1:
            bis.append((word[0], 'S'))
        else:
            for i,char in enumerate(word):
                if i==0:
                    bis.append((word[i], 'B'))
                else:
                    bis.append((word[i], 'I'))

    return bis


if __name__ == '__main__':
    convert2bis('分词\\train.txt', '分词\\bis.txt')


