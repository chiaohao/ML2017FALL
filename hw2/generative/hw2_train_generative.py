import numpy as np
import math
import sys

#normalize
def normalize(xs):
	xs_mean = np.mean(xs, axis=0)
	xs_std = np.std(xs, axis=0) * 100.0
	xs_norm = (xs - xs_mean) / xs_std
	return xs_norm, xs_mean, xs_std

#sigmoid function
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

#lose function
def Likeness(xs, ys, ws, b):
	acc = 0
	for i in range(len(ys)):
		y = 1 if sigmoid(np.sum(xs[i] * ws) + b) > 0.5 else 0
		if y == ys[i]:
			acc += 1
	return float(acc) / float(len(ys))
	#tmp = (1.0 - ys) * sigmoid(np.sum(xs * ws, axis=1) + b) + ys * (1.0 - sigmoid(np.sum(xs * ws, axis=1) + b))
	#print(tmp)
	#return np.prod(tmp)

#read xs, ys
xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
ys = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=1)
xs_test = np.genfromtxt(sys.argv[3], delimiter=',', skip_header=1)

#normalize xs
xs, mean, std = normalize(xs)
xs_test = (xs_test - mean) / std

#generative
count1 = 0
count2 = 0
mu1 = np.zeros(xs[0].shape)
mu2 = np.zeros(xs[0].shape)
for i in range(len(ys)):
	if ys[i] == 1:
		mu1 += xs[i]
		count1 += 1
	else:
		mu2 += xs[i]
		count2 += 1
mu1 /= float(count1)
mu2 /= float(count2)

sigma1 = np.zeros((xs[0].shape[0], xs[0].shape[0]))
sigma2 = np.zeros((xs[0].shape[0], xs[0].shape[0]))
for i in range(len(ys)):
        if ys[i] == 1:
            sigma1 += np.dot(np.transpose([xs[i] - mu1]), [(xs[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([xs[i] - mu2]), [(xs[i] - mu2)])
sigma1 /= count1
sigma2 /= count2
sigma_shared = (float(count1) / float(len(ys))) * sigma1 + (float(count2) / float(len(ys))) * sigma2

sigma_inverse = np.linalg.inv(sigma_shared)
w = np.dot( (mu1-mu2), sigma_inverse)
xs_test_t = xs_test.T
b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(count1)/float(count2))
a = np.dot(w, xs_test_t) + b
y = sigmoid(a)
y_ = np.around(y)
with open(sys.argv[4], 'w') as f:
	f.write('id,label\n' + '\n'.join(['%d,%d' %(index + 1, y) for index, y in enumerate(y_)]))