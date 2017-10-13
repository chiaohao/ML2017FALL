import numpy as np
import csv

#lose function
def Lose(ws, b, xs, ys):
	r = 0
	for index, x in enumerate(xs):
		r += (np.sum(np.multiply(ws, x)) + b - ys[index]) ** 2
	return np.sqrt(r/ys.shape[0])

#time data (18, n), n=20*24 for one set, n=9*24 for last set
data_csv = np.genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=range(3, 27))
data = np.reshape(data_csv, (data_csv.shape[0] // 18, 18, data_csv.shape[1]))
data[np.isnan(data)] = 0
data = np.transpose(data, (1,0,2))
data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

sTime = [5, 9]
useOnlyPM25 = [True, False]

for st in sTime:
	#get x and y inputs
	xs = None
	ys = None
	for i in range(12):
		seriesData = data[:, i*20*24:(i+1)*20*24]
		for j in range(seriesData.shape[1] - st):
			if xs is None:
				xs = seriesData[:, j:j+st].reshape(1, 18, st)
			else:
				xs = np.concatenate((xs, seriesData[:, j:j+st].reshape(1, 18, st)))
			if ys is None:
				ys = seriesData[9, j+st].reshape(1)
			else:
				ys = np.concatenate((ys, seriesData[9, j+st].reshape(1)))

	for uopm25 in useOnlyPM25:
		#initial point and learning rate
		ws = np.zeros(xs[0].shape)
		b = 0
		lr = 100
		iteration = 10000

		lr_b = 0
		lr_ws = np.zeros(xs[0].shape)
		#gradient decent
		lose = []
		for i in range(iteration):
			b_grad = 0.0
			ws_grad = np.zeros(xs[0].shape)

			b_grad = b_grad - np.sum(2.0 * (ys - b - np.sum(np.sum(xs * ws, axis=1), axis=1)) * 1.0)
			ws_grad = ws_grad - np.sum(2.0 * (ys - b - np.sum(np.sum(xs * ws, axis=1), axis=1)).repeat(18*st).reshape((ys.shape[0],18,st)) * xs, axis=0)

			lr_b = lr_b + b_grad ** 2
			lr_ws = lr_ws + ws_grad * ws_grad

			b = b - lr / np.sqrt(lr_b) * b_grad
			ws = ws - lr / np.sqrt(lr_ws) * ws_grad

			if uopm25:
				tmp = -ws
				tmp[9] = 0
				ws = ws + tmp

			if i % 50 == 49:
				lose.append([str(i + 1), str(Lose(ws, b, xs, ys))])

			if i % 1000 == 999:
				print ("iteration: " + str(i+1))
				print (Lose(ws, b, xs, ys))
		if uopm25:
			b.tofile('b_' + str(st) + 'hr_pm2.5.csv', sep=',')
			ws.tofile('ws_' + str(st) + 'hr_pm2.5.csv', sep=',')
			with open('lose_' + str(st) + 'hr_pm2.5.csv', 'w') as f:
				w = csv.writer(f)
				w.writerows(lose)
		else:
			b.tofile('b_' + str(st) + 'hr_all.csv', sep=',')
			ws.tofile('ws_' + str(st) + 'hr_all.csv', sep=',')
			with open('lose_' + str(st) + 'hr_all.csv', 'w') as f:
				w = csv.writer(f)
				w.writerows(lose)


