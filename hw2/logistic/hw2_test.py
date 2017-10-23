import numpy as np
import sys

b = np.genfromtxt('logistic/b.csv', delimiter=',')
ws = np.genfromtxt('logistic/ws.csv', delimiter=',')
xs_mean = np.genfromtxt('logistic/xs_mean.csv', delimiter=',')
xs_std = np.genfromtxt('logistic/xs_std.csv', delimiter=',')


testX = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
testX = (testX - xs_mean) / xs_std

#sigmoid function
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

output = sigmoid(np.sum(ws * testX, axis=1) + b)
#print (output)

outContent = []
outContent.append(['id', 'label'])
for i in range(output.shape[0]):
	o = 0
	if output[i] >= 0.5:
		o = 1
	else:
		o = 0
	outContent.append([str(i+1), str(o)])
with open(sys.argv[2], 'w') as file:
	file.write('\n'.join([','.join(o) for o in outContent]))