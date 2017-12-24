from skimage import data as sData
from skimage import io as sIo
import sys
import os
import numpy as np
from numpy.linalg import svd

data_path = sys.argv[1]
if data_path[len(data_path)-1] == '/':
	data_path = data_path[:len(data_path)-1]

data = []
for i in os.listdir(data_path):
	data.append(sIo.imread(data_path + '/' + i))

data = np.array(data)

avg = np.average(data, axis=0)
sIo.imsave('pca/avg.png', avg / 255.0)

data = data.reshape((data.shape[0], -1))
data = data / 255.0
for i in range(len(data)):
	data[i] -= np.average(data[i])

print(data.shape)
u, s, v = svd(data, full_matrices=False)
print(u.shape)
print(s.shape)
print(v.shape)

for i in range(4):
	t = v[i]
	t -= np.min(t)
	t /= np.max(t)
	sIo.imsave('pca/eigenfaces/eig%d.png' % i, t.reshape((600,600,3)))

	rate = s[i] / np.sum(s)
	print(rate)

_u = u[:,:4]
_s = s[:4]
_v = v[:4,:]

restruct_m = np.dot(_u * _s, _v)
for i in range(150,154):
	sIo.imsave('pca/reconstruction/reconstruct%d.png' % i, restruct_m[i].reshape((600,600,3)) / 255.0)