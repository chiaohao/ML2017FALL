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
    xs = xs / 255.0

with open(sys.argv[2]) as csvfile2:
    r = list(csv.reader(csvfile2))
    r = r[1:]
    xs_predict = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs_predict = xs_predict / 255.0

model = Sequential()
model.add(Dense(256, activation='selu', input_dim=48*48))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1024, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1024, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1024, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

csvLogger = CSVLogger('log_dnn.csv', append=True, separator=',')
cp = ModelCheckpoint("model_dnn.h5", monitor='val_loss', save_best_only=True)

model.fit(
        xs,
        ys,
        batch_size=64,
        epochs=300,
        validation_split=0.15,
        callbacks=[cp, csvLogger]
        )

plot_model(model, to_file='model_dnn_plot.png')

result = model.predict(xs_predict, batch_size=64)
result = [r.argmax() for r in result]

w = []
for idx, i in enumerate(result):
    w.append([str(idx), str(int(round(i)))])
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n' + '\n'.join([','.join(i) for i in w]))
