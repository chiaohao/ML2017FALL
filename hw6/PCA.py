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
for i in range(len(os.listdir(data_path))):
	if str(i) + '.jpg' in os.listdir(data_path):
		data.append(sIo.imread(data_path + '/' + str(i) + '.jpg'))

data = np.array(data, dtype=np.float64)

avg = np.average(data, axis=0)
sIo.imsave('pca/avg.png', avg / 255.0)

data -= avg
data = data.reshape((data.shape[0], -1))
data = data / 255.0
#for i in range(len(data)):
#	data[i] -= np.average(data[i])

print(data.shape)
u, s, v = svd(data, full_matrices=False)
print(u.shape)
print(s.shape)
print(v.shape)

for i in range(10):
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
	r = restruct_m[i].reshape((600,600,3)) + avg
	r -= np.min(r)
	r /= np.max(r)
	sIo.imsave('pca/reconstruction/reconstruct%d.png' % i, r)