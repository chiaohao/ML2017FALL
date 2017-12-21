import keras.models as kmodels
from keras.layers import Dense, Dropout, Embedding, Input, Flatten, Dot, Add, Lambda, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras import backend as K

import csv
import numpy as np
from numpy import genfromtxt
import sys
import pickle

def RMSE(y_true, y_pred):
    pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow((pred - y_true), 2)))

with open(sys.argv[1], 'r') as f:
    users_temp = f.readlines()
    users = [u.split('-')[0].replace('\n','').replace('M','1').replace('F','0').split('::') for u in users_temp][1:]
    users = sorted([[int(i) for i in u] for u in users])
    users = [u[1:] for u in users]
    users = np.array(users, dtype=np.int32)

with open(sys.argv[2], 'r', encoding='utf-8', errors='ignore') as f:
    movies_temp = f.readlines()
    movies = [m.replace('\n','').split('::') for m in movies_temp][1:]
    genres = []
    for m in movies:
        gs = m[2].split('|')
        for g in gs:
            if g not in genres:
                genres.append(g)
    genres = sorted(genres)
    tmp = np.zeros((len(movies), len(genres)))
    for i in range(len(movies)):
        for j in range(len(genres)):
            t = movies[i][2].split('|')
            if (genres[j] in t):
                tmp[i][j] = 1
    movies_genres = np.array(tmp)
    movies_max = np.max(np.array([int(m[0]) for m in movies]))

data = genfromtxt(sys.argv[3], delimiter=',', skip_header=True)
data_train = []
data_val = []
for i in range(len(data)):
    if i % 100 == 0:
        data_val.append(data[i])
    else:
        data_train.append(data[i])
data_train = np.array(data_train)
data_val = np.array(data_val)
u_xs = data_train[:,1]
m_xs = data_train[:,2]
ys = data_train[:,3]
u_xs_val = data_val[:,1]
m_xs_val = data_val[:,2]
ys_val = data_val[:,3]

M_input = Input(shape=[1])
M_vec = Flatten()(Embedding(movies_max,64,embeddings_initializer='random_normal')(M_input))
M_vec = Dropout(0.5)(M_vec)

U_input = Input(shape=[1])
U_vec = Flatten()(Embedding(len(users),64,embeddings_initializer='random_normal')(U_input))
U_vec = Dropout(0.5)(U_vec)

M_bias = Flatten()(Embedding(len(movies_genres),1,embeddings_initializer='zeros')(M_input))
U_bias = Flatten()(Embedding(len(users),1,embeddings_initializer='zeros')(U_input))

input_vecs = Dot(axes=1)([U_vec, M_vec])
input_vecs = Add()([input_vecs, U_bias, M_bias])

input_vecs = Lambda(lambda x: x + K.constant(3.5, dtype=K.floatx()))(input_vecs)

model = kmodels.Model([U_input, M_input], input_vecs)
model.compile(loss='mse', optimizer='adam', metrics=[RMSE])
model.summary()

csvLogger = CSVLogger('log.csv', append=True, separator=',')
cp = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
e = EarlyStopping(monitor='val_loss', patience=10)

model.fit([u_xs, m_xs], ys, batch_size=512, epochs=300, validation_data=([u_xs_val, m_xs_val], ys_val), callbacks=[csvLogger, cp, e])

