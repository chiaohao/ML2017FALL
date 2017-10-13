import numpy as np
import sys

b = np.genfromtxt(sys.argv[1], delimiter=',')
ws = np.genfromtxt(sys.argv[2], delimiter=',').reshape((18, 9))

test = np.genfromtxt(sys.argv[3], delimiter=',', usecols=range(2, 11))
test = np.reshape(test, (test.shape[0] // 18, 18, 9))
test[np.isnan(test)] = 0

output = b + np.sum(np.sum(ws * test, axis=1), axis=1)

outContent = []
outContent.append(['id', 'value'])
for i in range(output.shape[0]):
	outContent.append(['id_'+str(i), str(output[i])])
with open(sys.argv[4], 'w') as file:
	file.write('\n'.join([','.join(o) for o in outContent]))
