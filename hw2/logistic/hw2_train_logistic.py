import numpy as np
import math

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
xs = np.genfromtxt('data/X_train', delimiter=',', skip_header=1)
ys = np.genfromtxt('data/Y_train', delimiter=',', skip_header=1)

#normalize xs
xs, mean, std= normalize(xs)
mean.tofile('xs_mean.csv', sep=',')
std.tofile('xs_std.csv', sep=',')

#initial point and learning rate
ws = np.zeros(xs[0].shape)
b = 0.0
lr = 100.0
iteration = 100000

lr_b = 0.0
lr_ws = np.zeros(xs[0].shape)

#gradient decent
for i in range(iteration):
	b_grad = 0.0
	ws_grad = np.zeros(xs[0].shape)

	b_grad = b_grad - np.sum(ys - sigmoid(np.sum(xs * ws, axis=1) + b) * 1.0)
	ws_grad = ws_grad - np.sum((ys - sigmoid(np.sum(xs * ws, axis=1) + b)) * xs.T, axis=1)
	
	lr_b = lr_b + b_grad ** 2
	lr_ws = lr_ws + ws_grad * ws_grad

	b = b - lr / np.sqrt(lr_b) * b_grad
	ws = ws - lr / np.sqrt(lr_ws) * ws_grad
	
	if i % 1000 == 999:
		print ("iteration: " + str(i+1))
		print (Likeness(xs, ys, ws, b))
		b.tofile('b.csv', sep=',')
		ws.tofile('ws.csv', sep=',')
