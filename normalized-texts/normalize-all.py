# -*- coding: utf-8 -*-

# imports
import pymystem3
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import pickle

data = 'after anna childhood dead evil three war'.split(' ')

stem = pymystem3.Mystem()


def tokenize_ru_and_normalize(file_text):
    # firstly let's apply nltk tokenization
    tokens = stem.lemmatize(file_text)

    print('Zero is done')

    # let's delete punctuation symbols
    tokens = [i for i in tokens if (i not in string.punctuation)]

    # deleting stop_words
    stop_words = stopwords.words('russian')
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', '–', 'к', 'на', '...', '…'])
    tokens = [i for i in tokens if (i not in stop_words)]

    # cleaning words
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]

    return tokens


def sentences_ru(text):
    return [tokenize_ru_and_normalize(sent) for sent in sent_tokenize(text, 'russian')]


def read(path):
    return open(path, 'r', encoding='utf-8').read()


for pah in data:
    pickle.dump(sentences_ru(read('./../big-model-III/' + pah + '.txt')), open('./' + pah + '.n', 'wb'))
    print('One is done')
