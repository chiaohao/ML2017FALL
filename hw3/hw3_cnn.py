from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

import csv
import numpy as np
import sys

with open(sys.argv[1]) as csvfile:
    r = list(csv.reader(csvfile))
    r = r[1:]
    ys = np_utils.to_categorical([int(i[0]) for i in r])
    xs = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs = xs.reshape(-1,48,48,1) / 255.0
    d = int(xs.shape[0] * 0.15)
    xs_train = xs[:-d]
    ys_train = ys[:-d]
    xs_test = xs[-d:]
    ys_test = ys[-d:]
'''
with open(sys.argv[2]) as csvfile2:
    r = list(csv.reader(csvfile2))
    r = r[1:]
    xs_predict = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs_predict = xs_predict.reshape(-1,48,48,1) / 255.0
'''
model = Sequential()

model.add(Convolution2D(64,3,3,input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Convolution2D(256,3,3))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Convolution2D(512,3,3))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Convolution2D(512,3,3))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

csvLogger = CSVLogger('log_cnn.csv', append=True, separator=',')
cp = ModelCheckpoint("model_cnn.h5", monitor='val_loss', save_best_only=True)
#e = EarlyStopping(monitor='val_loss', patience=100)
image_gen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
image_gen.fit(xs_train)
model.fit_generator(
        image_gen.flow(xs_train, ys_train, batch_size=64),
        steps_per_epoch=len(xs_train)//64,
        validation_data=(xs_test, ys_test),
        epochs=300,
        callbacks=[cp, csvLogger])

#plot_model(model, to_file='model_preimage_plot.png')
'''
result = model.predict(xs_predict, batch_size=64)
result = [r.argmax() for r in result]

w = []
for idx, i in enumerate(result):
    w.append([str(idx), str(int(round(i)))])
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n' + '\n'.join([','.join(i) for i in w]))
''' 
