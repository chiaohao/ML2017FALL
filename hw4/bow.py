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
#xs_nolabel = p[len(sentences1):]
tokenizer.fit_on_texts(sentences1)
xs = tokenizer.sequences_to_matrix(tokenizer.texts_to_sequences(sentences1))
#xs_nolabel = pad_sequences(tokenizer.texts_to_sequences(sentences2), maxlen=50)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

d = int(xs.shape[0] * 0.15)
xs_train = xs[:-d]
ys_train = ys[:-d]
xs_val = xs[-d:]
ys_val = ys[-d:]

model = Sequential()
#model.add(Embedding(max_words, 128, input_length=xs.shape[1]))
model.add(Dense(256, input_shape=xs_train[0].shape))
model.add(Dropout(0.3))
#model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

csvLogger = CSVLogger('log_rnn.csv', append=False, separator=',')
#cp = ModelCheckpoint("model_dnn.h5", monitor='val_loss', save_best_only=True)
#e = EarlyStopping(monitor='val_loss', patience=100)

print(xs_train.shape)
print(ys_train.shape)
cp = ModelCheckpoint('model_bow.h5', monitor='val_loss', save_best_only=True)
model.fit(xs_train, ys_train, batch_size=256, epochs=30, validation_data=(xs_val, ys_val), callbacks=[csvLogger, cp])

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
