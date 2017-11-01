from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils

import csv
import numpy as np
import sys

xs_train = []
ys_train = []
xs_test = []
ys_test = []
xs_predict = []
with open(sys.argv[1]) as csvfile:
    r = list(csv.reader(csvfile))
    r = r[1:]
    ys = np_utils.to_categorical([int(i[0]) for i in r])
    xs = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs = xs.reshape(-1,48,48,1) / 255.0
    n = int(float(ys.shape[0]) * 0.7)
    xs_train = xs[:n]
    xs_test = xs[n:]
    ys_train = ys[:n]
    ys_test = ys[n:]

with open(sys.argv[2]) as csvfile2:
    r = list(csv.reader(csvfile2))
    r = r[1:]
    xs_predict = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs_predict = xs_predict.reshape(-1,48,48,1) / 255.0

model = Sequential()

model.add(Convolution2D(50,4,4,input_shape=(48,48,1), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(100,4,4, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(200,4,4, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(800, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=xs_train, y=ys_train, batch_size=200, epochs=50)

score = model.evaluate(xs_train, ys_train)
print('\nTraining Set')
print('Loss: ', score[0])
print('Accuracy: ', score[1])

score = model.evaluate(xs_test, ys_test)
print('\nTesting Set')
print('Loss: ', score[0])
print('Accuracy: ', score[1])

result = model.predict(xs_predict)
result = [r.argmax() for r in result]

w = []
for idx, i in enumerate(result):
    w.append([str(idx), str(int(round(i)))])
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n' + '\n'.join([','.join(i) for i in w]))
