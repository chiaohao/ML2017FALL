from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Embedding

from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import csv
import numpy as np
import sys
import codecs
import pickle

max_words = 20000
tokenizer = Tokenizer(num_words=max_words)

with codecs.open(sys.argv[1], 'r', encoding='utf-8') as file1:
    r = file1.readlines()
    s = [i.split(' +++$+++ ') for i in r]
    ys = np_utils.to_categorical([int(i[0]) for i in s])
    sentences1 = [i[1] for i in s]

with codecs.open(sys.argv[2], 'r', encoding='utf-8') as file2:
    sentences2 = file2.readlines()

tokenizer.fit_on_texts(sentences1 + sentences2)
p = pad_sequences(tokenizer.texts_to_sequences(sentences1 + sentences2))
xs = p[:len(sentences1)]
xs_nolabel = p[len(sentences1):]

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    d = int(xs.shape[0] * 0.15)
    xs_train = xs[:-d]
    ys_train = ys[:-d]
    xs_test = xs[-d:]
    ys_test = ys[-d:]

model = Sequential()
model.add(Embedding(max_words, 128, input_length=xs.shape[1]))
model.add(LSTM(256))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

csvLogger = CSVLogger('log_dnn.csv', append=True, separator=',')
cp = ModelCheckpoint("model_dnn.h5", monitor='val_loss', save_best_only=True)
#e = EarlyStopping(monitor='val_loss', patience=100)

for i in range(5):
    model.fit(xs, ys, batch_size=2048, epochs=2, validation_split=0.15, callbacks=[csvLogger, cp])
    ys_nolabel = model.predict(xs_nolabel, batch_size=2048)
    xs_toAdd = []
    ys_toAdd = []
    xs_notAdd = []
    for j in range(len(xs_nolabel)):
        if ys_nolabel[j][0] > 0.9 or ys_nolabel[j][1] > 0.9:
            ys_toAdd.append([round(k) for k in ys_nolabel[j]])
            xs_toAdd.append(xs_nolabel[j])
        else:
            xs_notAdd.append(xs_nolabel[j])
    xs = np.concatenate((xs, xs_toAdd), axis=0)
    ys = np.concatenate((ys, ys_toAdd), axis=0)
    xs_nolabel = xs_notAdd

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
