from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Embedding

from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import csv
import numpy as np
import sys
import codecs
import pickle

max_words = 20000
tokenizer = Tokenizer(num_words=max_words, filters='')

with codecs.open(sys.argv[1], 'r', encoding='utf-8') as file:
    r = file.readlines()
    s = [i.split(' +++$+++ ') for i in r]
    ys = np.array([[int(i[0])] for i in s])
    sentences = [i[1] for i in s]

tokenizer.fit_on_texts(sentences)
xs = pad_sequences(tokenizer.texts_to_sequences(sentences), maxlen=37)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

fold_num = 10
xs_train_folds = []
ys_train_folds = []

for i in range(fold_num):
    n = (int)(xs.shape[0] / fold_num)
    xs_train_folds.append(xs[i*n:(i+1)*n])
    ys_train_folds.append(ys[i*n:(i+1)*n])

for i in range(fold_num):
    xs_train = []
    ys_train = []
    xs_val = []
    ys_val = []
    for j in range(fold_num):
        if i == j:
            xs_val = xs_train_folds[j]
            ys_val = ys_train_folds[j]
            continue
        if xs_train == []:
            xs_train = xs_train_folds[j]
            ys_train = ys_train_folds[j]
        else:
            xs_train = np.concatenate((xs_train, xs_train_folds[j]), axis=0)
            ys_train = np.concatenate((ys_train, ys_train_folds[j]), axis=0)
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=xs.shape[1]))
    model.add(LSTM(256, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.summary()

    csvLogger = CSVLogger('log_rnn.csv', append=True, separator=',')
    cp = ModelCheckpoint("kfold/model_rnn%d.h5" % (i), monitor='val_acc', save_best_only=True)
    e = EarlyStopping(monitor='val_acc', patience=3)
    
    model.fit(xs_train, ys_train, batch_size=256, epochs=20, validation_data=(xs_val, ys_val), callbacks=[csvLogger, cp, e])

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
