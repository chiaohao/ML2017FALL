import jieba
from gensim.models.word2vec import Word2Vec
import sys
import operator
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
zhfont1 = FontProperties(fname='msjh.ttf')
from matplotlib import pyplot as plt
from adjustText import adjust_text
from sklearn.manifold import TSNE

jieba.set_dictionary('dict.txt.big.txt')

with open(sys.argv[1], 'r', encoding='utf-8') as f:
	data = [i.replace('\n', '') for i in f.readlines()]

data_cut = [list(jieba.cut(i)) for i in data]

word_count = dict()
for line in data_cut:
	for word in line:
		if word not in word_count:
			word_count[word] = 1
		else:
			word_count[word] = word_count[word] + 1

sorted_word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)

c = 0
for t in sorted_word_count:
	if(t[1] < 3500):
		break
	c += 1

sorted_word_count = sorted_word_count[:c]
labels = [i[0] for i in sorted_word_count]

w2v = Word2Vec(data_cut, size=200)
w2v.wv.save_word2vec_format('w2v.txt', binary=False)

f = open('w2v.txt', encoding='utf-8')
embeddings_index = dict()
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(labels), 100))
for i, word in enumerate(labels):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

vis_data = TSNE(n_components=2).fit_transform(embedding_matrix)
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]

for i in range(len(vis_x)):
    plt.scatter(vis_x[i],vis_y[i], color='blue', marker='.', linewidths=0)

texts = []
for x, y, s in zip(vis_x, vis_y, labels):
    texts.append(plt.text(x, y, s, size=7, fontproperties=zhfont1))
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
'''
    plt.annotate(labels[i], 
    	xy=(vis_x[i], vis_y[i]), 
    	xytext=(5, 2),
    	textcoords='offset points',
    	ha='right',
    	va='bottom')
'''
fig = plt.gcf()
fig.savefig('word_embedding.png', dpi=200)