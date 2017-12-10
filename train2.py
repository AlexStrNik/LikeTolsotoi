# -*- coding: utf-8 -*-

# imports
import gensim
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# load text
text = open('./war.txt', 'r', encoding='utf-8').read()


def tokenize_ru(file_text):
    # firstly let's apply nltk tokenization
    tokens = word_tokenize(file_text)

    # let's delete punctuation symbols
    tokens = [i for i in tokens if (i not in string.punctuation)]

    # deleting stop_words
    stop_words = stopwords.words('russian')
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', '–', 'к', 'на', '...', '…'])
    tokens = [i for i in tokens if (i not in stop_words)]

    # cleaning words
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]

    return tokens


# tokenize sentences
sentences = [tokenize_ru(sent) for sent in sent_tokenize(text, 'russian')]
print(len(sentences))  # 30938
print(sentences[200:209])  # [['Он', 'нагнув', 'голову', 'расставив', 'большие', 'ноги', 'стал', 'доказывать', 'Анне', 'Павловне', 'почему', 'полагал', 'план', 'аббата', 'химера'],...]

# train model part II
model = gensim.models.Word2Vec.load('./w2v.model')
model.train(sentences, total_examples=int(model.corpus_count*1.2), epochs=int(model.iter*1.2))

# save model
model.save('./w2v-II.model')
print('saved')
