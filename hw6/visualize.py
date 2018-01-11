from keras.models import load_model
import sys
import numpy as np
from sklearn.cluster import KMeans
from numpy import genfromtxt

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

model = load_model(sys.argv[1])

data = np.load(sys.argv[2]).reshape((-1,28,28,1)) / 255.

data_encode = model.predict(data)
data_encode = data_encode.reshape((data_encode.shape[0], -1))

vis_data = TSNE(n_components=2, perplexity=10).fit_transform(data_encode)
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]

# my predict label
kmeans = KMeans(n_clusters=2).fit(data_encode)
result = kmeans.labels_
c1 = []
c2 = []
for i in range(len(result)):
	if result[i] == 0:
		c1.append([vis_x[i], vis_y[i]])
	else:
		c2.append([vis_x[i], vis_y[i]])
c1 = np.array(c1)
c2 = np.array(c2)

print(len(c1))
print(len(c2))

colors = ['b', 'r']
sc0 = plt.scatter(c1[:,0], c1[:,1], marker='.', color=colors[0], alpha=0.7, edgecolors='none')
sc1 = plt.scatter(c2[:,0], c2[:,1], marker='.', color=colors[1], alpha=0.7, edgecolors='none')

plt.legend((sc0, sc1), iter(['Class1', 'Class2']), loc='lower left', ncol=1, fontsize=8)
plt.show()
fig = plt.gcf()
fig.savefig('visualize_predict.png', dpi=200)

#true label
colors = ['b', 'r']
sc0 = plt.scatter(vis_x[:5000], vis_y[:5000], marker='.', color=colors[0], alpha=0.7, edgecolors='none')
sc1 = plt.scatter(vis_x[5000:], vis_y[5000:], marker='.', color=colors[1], alpha=0.7, edgecolors='none')

plt.legend((sc0, sc1), iter(['Class1', 'Class2']), loc='lower left', ncol=1, fontsize=8)
plt.show()
fig = plt.gcf()
fig.savefig('visualize_true.png', dpi=200)