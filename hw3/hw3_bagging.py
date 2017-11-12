from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from sklearn.cross_validation import KFold

import csv
import numpy as np
import sys

def batch_generator(x, y, batch_size):
    batches_number = mp.ceil(x.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(x.shape[0])
    np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter : batch_size * (counter + 1)]
        x_batch = x[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield x_batch, y_batch
        if(counter == batches_number):
            np.random.shuffle(sample_index)
            counter = 0

def batch_generator_predict(x, batch_size):
    batches_number = mp.ceil(x.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(x.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter : batch_size * (counter + 1)]
        x_batch = x[batch_index,:].toarray()
        counter += 1
        yield x_batch
        if(counter == batches_number):
            counter = 0

with open(sys.argv[1]) as csvfile:
    r = list(csv.reader(csvfile))
    r = r[1:]
    ys = np_utils.to_categorical([int(i[0]) for i in r])
    xs = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs = xs.reshape(-1,48,48,1) / 255.0
    d = int(xs.shape[0] * 0.1)
    xs_train = xs[:-d]
    ys_train = ys[:-d]
    xs_valid = xs[-d:]
    ys_valid = ys[-d:]

with open(sys.argv[2]) as csvfile2:
    r = list(csv.reader(csvfile2))
    r = r[1:]
    xs_predict = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs_predict = xs_predict.reshape(-1,48,48,1) / 255.0

model = Sequential()

model.add(Convolution2D(64,3,3,input_shape=(48,48,1)))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(128,3,3))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(256,3,3))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(512,3,3))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

e = EarlyStopping(monitor='val_loss', patience=10)
tr_g = batch_generator(xs_train, ys_train, 128)
tv_g = batch_generator(xs_valid, ys_valid, 128)
p_g = batch_generatorp(xs_predict, xs_predict.shape[0])

#history = model.fit(x=xs, y=ys, validation_split=0.2, batch_size=128, epochs=1000, callbacks=[e])

nfolds = 10
folds = KFold(len(ys), n_folds=nfolds, shuffle=True, random_state=123)

ii = 0
nbags = 10
epochs = 50

##############################

model.save('model_bagging_save.h5')
plot_model(model, to_file='model_bagging_plot.png')

result = model.predict(xs_predict, batch_size=128)
result = [r.argmax() for r in result]

w = []
for idx, i in enumerate(result):
    w.append([str(idx), str(int(round(i)))])
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n' + '\n'.join([','.join(i) for i in w]))
