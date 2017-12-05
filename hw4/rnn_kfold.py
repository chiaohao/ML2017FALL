from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Embedding, Conv1D

from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec

import csv
import numpy as np
import sys
import codecs
import pickle

tokenizer = Tokenizer(filters='\t\n')

with codecs.open(sys.argv[1], 'r', encoding='utf-8') as file:
    r = file.readlines()
    s = [i.split(' +++$+++ ') for i in r]
    ys = np.array([[int(i[0])] for i in s])
    sentences = [i[1] for i in s]

with codecs.open(sys.argv[2], 'r', encoding='utf-8') as file:
    sentences_n = file.readlines()

sentences_n.extend(sentences)
tokenizer.fit_on_texts(sentences_n)
xs = pad_sequences(tokenizer.texts_to_sequences(sentences), maxlen=37)

with open('tokenizer_.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

fold_num = 10
xs_train_folds = []
ys_train_folds = []

for i in range(fold_num):
    n = (int)(xs.shape[0] / fold_num)
    xs_train_folds.append(xs[i*n:(i+1)*n])
    ys_train_folds.append(ys[i*n:(i+1)*n])

w2v = Word2Vec([list(filter(None, l.split())) for l in sentences_n], size=200, workers=1)
w2v.wv.save_word2vec_format('w2v.txt', binary=False)
f = open('w2v.txt', encoding='utf-8')
embeddings_index = dict()
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(tokenizer.word_index), 200))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

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
    model.add(Embedding(len(tokenizer.word_index), 200, weights=[embedding_matrix], input_length=xs.shape[1], trainable=False))
#    model.add(Conv1D(512, 3, activation='relu'))
#    model.add(Conv1D(256, 3, activation='relu'))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.summary()

    csvLogger = CSVLogger('log_rnn.csv', append=True, separator=',')
    cp = ModelCheckpoint("kfold/model_rnn%d.h5" % (i), monitor='val_acc', save_best_only=True)
    e = EarlyStopping(monitor='val_acc', patience=5)
    
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
