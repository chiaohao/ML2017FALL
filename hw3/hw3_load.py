from keras.models import load_model
import sys
import csv
import numpy as np

model = load_model(sys.argv[1])

with open(sys.argv[2]) as csvfile2:
    r = list(csv.reader(csvfile2))
    r = r[1:]
    xs_predict = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs_predict = xs_predict.reshape(-1,48,48,1) / 255.0

result = model.predict(xs_predict, batch_size=64)
result = [r.argmax() for r in result]

w = []
for idx, i in enumerate(result):
    w.append([str(idx), str(int(round(i)))])
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n' + '\n'.join([','.join(i) for i in w]))
