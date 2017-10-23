import numpy as np
import sys
from sklearn.ensemble import GradientBoostingClassifier

#read xs, ys
xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
ys = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=1)
xs_test = np.genfromtxt(sys.argv[3], delimiter=',', skip_header=1)

l = GradientBoostingClassifier(max_depth=5).fit(xs, ys)
output = l.predict(xs_test)

with open(sys.argv[4], 'w') as f:
	f.write('id,label\n' + '\n'.join(['%d,%d' %(index + 1, y) for index, y in enumerate(output)]))