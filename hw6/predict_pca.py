import sys
import numpy as np
from sklearn.cluster import KMeans
from numpy import genfromtxt
from sklearn.decomposition import PCA

data = np.load(sys.argv[1]) / 255.

pca = PCA(n_components=784, svd_solver='randomized', whiten=True).fit_transform(data)

kmeans = KMeans(n_clusters=2).fit(pca)
result = kmeans.labels_

testing_data = genfromtxt(sys.argv[2], delimiter=',', skip_header=1, dtype=int)
testing_data = testing_data[:,1:]
#print(testing_data[0])

output = [1 if result[i[0]] == result[i[1]] else 0 for i in testing_data]
with open(sys.argv[3], 'w') as f:
    f.write('ID,Ans\n' + '\n'.join([','.join([str(idx), str(o)]) for idx, o in enumerate(output)]))

#print(result)
print(np.count_nonzero(result))
print(len(result) - np.count_nonzero(result))