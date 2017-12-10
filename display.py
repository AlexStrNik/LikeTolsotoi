# -*- coding: utf-8 -*-

# imports
import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

model = gensim.models.Word2Vec.load('./w2v-II.model')

print(len(model.wv.vocab))

vocab = list(model.wv.vocab)[:200]
X = model[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df = pd.concat([pd.DataFrame(X_tsne),
                pd.Series(vocab)],
               axis=1)

df.columns = ['x', 'y', 'word']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for i, txt in enumerate(df['word']):
    print(i)
    ax.annotate(txt, (df['x'].iloc[i], df['y'].iloc[i]))

plt.show()
