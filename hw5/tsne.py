from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import sys
import csv
import numpy as np
from numpy import genfromtxt

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def RMSE(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

model = load_model(sys.argv[1], custom_objects={'RMSE': RMSE})

weights = np.array(model.get_layer(name="embedding_1").get_weights()).squeeze()
#print(weights.shape)
#np.save("movie_emb.npy")

with open(sys.argv[2], 'r', encoding='utf-8', errors='ignore') as f:
    movies_temp = f.readlines()
    movies = [m.replace('\n','').split('::') for m in movies_temp][1:]
    x = [int(m[0]) for m in movies]

    for i in range(len(movies)):
    	for j in ['Action', 'Adventure', 'War', 'Documentary', 'Western']:
    		movies[i][2] = movies[i][2].replace(j, 'Action')
    	for j in ['Crime', 'Film-Noir', 'Horror', 'Thriller']:
    		movies[i][2] = movies[i][2].replace(j, 'Crime')
    	for j in ['Comedy', 'Drama', 'Musical', 'Romance']:
    		movies[i][2] = movies[i][2].replace(j, 'Comedy')
    	for j in ['Fantasy', 'Mystery', 'Sci-Fi']:
    		movies[i][2] = movies[i][2].replace(j, 'Fantasy')
    	for j in ['Animation', "Children's"]:
    		movies[i][2] = movies[i][2].replace(j, 'Animation')

    genres = []
    for m in movies:
        gs = m[2].split('|')
        for g in gs:
            if g not in genres:
                genres.append(g)
    genres = sorted(genres)
    tmp = np.zeros((len(movies), len(genres)))
    for i in range(len(movies)):
        for j in range(len(genres)):
            t = movies[i][2].split('|')
            if (genres[j] in t):
                tmp[i][j] = 1
    movies_genres = np.array(tmp)
    print(genres)
    y = np.argmax(movies_genres, axis=1)
    x = np.array(weights)

    vis_data = TSNE(n_components=2, perplexity=10).fit_transform(x)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]

    data = []
    for i in range(5):
    	data.append([[], []])
    for i in range(len(y)):
    	data[y[i]][0].append(vis_x[i])
    	data[y[i]][1].append(vis_y[i])
    data = np.array(data)

    colors = ['b', 'c', 'y', 'm', 'r']
    sc0 = plt.scatter(data[0][0], data[0][1], marker='.', color=colors[0], alpha=0.7, edgecolors='none')
    sc1 = plt.scatter(data[1][0], data[1][1], marker='.', color=colors[1], alpha=0.7, edgecolors='none')
    sc2 = plt.scatter(data[2][0], data[2][1], marker='.', color=colors[2], alpha=0.7, edgecolors='none')
    sc3 = plt.scatter(data[3][0], data[3][1], marker='.', color=colors[3], alpha=0.7, edgecolors='none')
    sc4 = plt.scatter(data[4][0], data[4][1], marker='.', color=colors[4], alpha=0.7, edgecolors='none')

    #cm = plt.cm.get_cmap('Set1')
    #sc = plt.scatter(vis_x, vis_y, marker='.', c=y, cmap = cm)
    #plt.colorbar(sc)
    plt.legend((sc0, sc1, sc2, sc3, sc4),
           iter(['Class1', 'Class2', 'Class3', 'Class4', 'Class5']),
           loc='lower left',
           ncol=1,
           fontsize=8)
    plt.show()
    fig = plt.gcf()
    fig.savefig('tsne.png', dpi=200)
    
