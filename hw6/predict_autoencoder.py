from keras.models import load_model
import sys
import numpy as np
from sklearn.cluster import KMeans
from numpy import genfromtxt

model = load_model(sys.argv[1])

data = np.load(sys.argv[2]).reshape((-1,28,28,1)) / 255.

data_encode = model.predict(data)
data_encode = data_encode.reshape((data_encode.shape[0], -1))
#print(data_encode.shape)

kmeans = KMeans(n_clusters=2).fit(data_encode)
result = kmeans.labels_

testing_data = genfromtxt(sys.argv[3], delimiter=',', skip_header=1, dtype=int)
testing_data = testing_data[:,1:]
#print(testing_data[0])

output = [1 if result[i[0]] == result[i[1]] else 0 for i in testing_data]
with open(sys.argv[4], 'w') as f:
    f.write('ID,Ans\n' + '\n'.join([','.join([str(idx), str(o)]) for idx, o in enumerate(output)]))

#print(result)
print(np.count_nonzero(result))
print(len(result) - np.count_nonzero(result))
