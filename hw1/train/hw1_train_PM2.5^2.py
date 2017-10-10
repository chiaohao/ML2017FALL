import numpy as np

#lose function
def Lose(ws, w2s, b, xs, ys):
	r = 0
	for index, x in enumerate(xs):
		r += (np.sum(np.multiply(ws, x)) + np.sum(np.multiply(np.multiply(w2s, x), x)) + b - ys[index]) ** 2
	return np.sqrt(r/ys.shape[0])


#time data (18, n), n=20*24 for one set, n=9*24 for last set
data_csv = np.genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=range(3, 27))
data = np.reshape(data_csv, (data_csv.shape[0] // 18, 18, data_csv.shape[1]))
data[np.isnan(data)] = 0
data = np.transpose(data, (1,0,2))
data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

#get x and y inputs
xs = None
ys = None
for i in range(12):
	seriesData = data[:, i*20*24:(i+1)*20*24]
	for j in range(seriesData.shape[1] - 9):
		if xs is None:
			xs = seriesData[:, j:j+9].reshape(1, 18, 9)
		else:
			xs = np.concatenate((xs, seriesData[:, j:j+9].reshape(1, 18, 9)))
		if ys is None:
			ys = seriesData[9, j+9].reshape(1)
		else:
			ys = np.concatenate((ys, seriesData[9, j+9].reshape(1)))

#initial point and learning rate
ws = np.zeros(xs[0].shape)
w2s = np.zeros(xs[0].shape)
b = 0
lr = 100
iteration = 200000

lr_b = 0
lr_ws = np.zeros(xs[0].shape)
lr_w2s = np.zeros(xs[0].shape)
#gradient decent
for i in range(iteration):
	b_grad = 0.0
	ws_grad = np.zeros(xs[0].shape)
	w2s_grad = np.zeros(xs[0].shape)

	b_grad = b_grad - np.sum(2.0 * (ys - b - np.sum(np.sum(xs * ws, axis=1), axis=1) - np.sum(np.sum(w2s * xs * xs, axis=1), axis=1)) * 1.0)
	ws_grad = ws_grad - np.sum(2.0 * (ys - b - np.sum(np.sum(xs * ws, axis=1), axis=1) - np.sum(np.sum(w2s * xs * xs, axis=1), axis=1)).repeat(18*9).reshape((ys.shape[0],18,9)) * xs, axis=0)
	w2s_grad = w2s_grad - np.sum(2.0 * (ys - b - np.sum(np.sum(xs * ws, axis=1), axis=1) - np.sum(np.sum(w2s * xs * xs, axis=1), axis=1)).repeat(18*9).reshape((ys.shape[0],18,9)) * xs * xs, axis=0)

	"""
	for j in range(xs.shape[0]):
		b_grad = b_grad - 2.0 * (ys[j] - b - np.sum(xs[j] * ws) - np.sum(w2s * xs[j] * xs[j])) * 1.0
		ws_grad = ws_grad - 2.0 * (ys[j] - b - np.sum(xs[j] * ws) - np.sum(w2s * xs[j] * xs[j])) * xs[j]
		w2s_grad = w2s_grad - 2.0 * (ys[j] - b - np.sum(xs[j] * ws) - np.sum(w2s * xs[j] * xs[j])) * (xs[j] * xs[j])
	"""

	lr_b = lr_b + b_grad ** 2
	lr_ws = lr_ws + ws_grad * ws_grad
	lr_w2s = lr_w2s + w2s_grad * w2s_grad

	b = b - lr / np.sqrt(lr_b) * b_grad
	ws = ws - lr / np.sqrt(lr_ws) * ws_grad
	w2s = w2s - lr / np.sqrt(lr_w2s) * w2s_grad

	tmp = -ws
	tmp[7,:] = 0
	tmp[8,:] = 0
	tmp[9,:] = 0
	tmp[12,:] = 0
	tmp[14,:] = 0
	tmp[15,:] = 0
	tmp[16,:] = 0
	tmp[17,:] = 0
	ws = ws + tmp

	tmp = -w2s
	tmp[8,:] = 0
	tmp[9,:] = 0
	w2s = w2s + tmp

	if i % 1000 == 999:
		print ("iteration: " + str(i+1))
		print (Lose(ws, w2s, b, xs, ys))
		b.tofile('b.csv', sep=',')
		ws.tofile('ws.csv', sep=',')
		w2s.tofile('w2s.csv', sep=',')


