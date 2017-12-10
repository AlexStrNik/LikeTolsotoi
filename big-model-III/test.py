# -*- coding: utf-8 -*-

# imports
import gensim

# load model
model = gensim.models.Word2Vec.load('./w2v-III.model')
print(len(model.wv.vocab))
