from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

import csv
import numpy as np
import sys

xs = []
ys = []
xs_test = []
with open(sys.argv[1]) as csvfile:
    r = list(csv.reader(csvfile))
    r = r[1:]
    ys_int = [int(i[0]) for i in r]
    ys = []
    for y in ys_int:
        ys.append([])
        c = len(ys) - 1
        for i in range(7):
            if y == i:
                ys[c].append(1)
            else:
                ys[c].append(0)
    xs = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs = xs.reshape(-1,48,48,1) / 255.0

with open(sys.argv[2]) as csvfile2:
    r = list(csv.reader(csvfile2))
    r = r[1:]
    xs_test = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs_test = xs_test.reshape(-1,48,48,1)

model = Sequential()
model.add(Convolution2D(52,5,5,input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(104,5,5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xs, ys, batch_size=52, epochs=20)
score = model.evaluate(xs, ys)
print('Loss: ', score[0])
print('Accuracy: ', score[1])

result = model.predict(xs_test)
result = np.sum(result * np.array([0,1,2,3,4,5,6]), axis=1)

w = []
for idx, i in enumerate(result):
    w.append([str(idx), str(int(round(i)))])
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n' + '\n'.join([','.join(i) for i in w]))
