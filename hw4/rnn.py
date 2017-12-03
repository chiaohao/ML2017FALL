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
tokenizer = Tokenizer(num_words=max_words)

with codecs.open(sys.argv[1], 'r', encoding='utf-8') as file1:
    r = file1.readlines()
    s = [i.split(' +++$+++ ') for i in r]
    ys = np.array([[int(i[0])] for i in s])
    sentences1 = [i[1] for i in s]

with codecs.open(sys.argv[2], 'r', encoding='utf-8') as file2:
    sentences2 = file2.readlines()

tokenizer.fit_on_texts(sentences1 + sentences2)
#p = pad_sequences(tokenizer.texts_to_sequences(sentences1 + sentences2), maxlen=100)
#xs = p[:len(sentences1)]
#xs_nolabel = p[len(sentences1):]
xs = pad_sequences(tokenizer.texts_to_sequences(sentences1), maxlen=50)
xs_nolabel = pad_sequences(tokenizer.texts_to_sequences(sentences2), maxlen=50)
print(xs.shape)
print(xs_nolabel.shape)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

d = int(xs.shape[0] * 0.15)
xs_train = xs[:-d]
ys_train = ys[:-d]
xs_val = xs[-d:]
ys_val = ys[-d:]

model = Sequential()
model.add(Embedding(max_words, 128, input_length=xs.shape[1]))
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

csvLogger = CSVLogger('log_rnn.csv', append=False, separator=',')
#cp = ModelCheckpoint("model_dnn.h5", monitor='val_loss', save_best_only=True)
#e = EarlyStopping(monitor='val_loss', patience=100)

for i in range(20):
    cp = ModelCheckpoint('model_rnn%d.h5' % (i), monitor='val_loss', save_best_only=True)
    model.fit(xs_train, ys_train, batch_size=256, epochs=1, validation_data=(xs_val, ys_val), callbacks=[csvLogger, cp])
    ys_nolabel = model.predict(xs_nolabel, batch_size=256)
    ys_nolabel = ys_nolabel.reshape((ys_nolabel.shape[0],1))
    xs_toAdd = []
    ys_toAdd = []
    xs_notAdd = []
    for j in range(len(xs_nolabel)):
        yn = ys_nolabel[j]
        if (yn > 0.75 and yn < 0.9) or (yn > 0.1 and yn < 0.25):
            ys_toAdd.append(np.around(ys_nolabel[j]))
            xs_toAdd.append(xs_nolabel[j])
        else:
            xs_notAdd.append(xs_nolabel[j])
    xs_train = np.concatenate((xs_train, np.array(xs_toAdd)), axis=0)
    ys_train = np.concatenate((ys_train, np.array(ys_toAdd)), axis=0)
    xs_nolabel = np.array(xs_notAdd)

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
