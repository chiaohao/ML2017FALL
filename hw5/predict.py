from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import sys
import csv
import numpy as np
from numpy import genfromtxt

def RMSE(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

model = load_model(sys.argv[1], custom_objects={'RMSE': RMSE})

data = genfromtxt(sys.argv[2], delimiter=',', skip_header=True)
u_xs = data[:,1]
m_xs = data[:,2]

'''
with open(sys.argv[2], 'r') as file:
    r = [i.replace('\n', '').split(',') for i in file.readlines()][1:]
    r = [i[1:] for i in r]
    u_xs = np.array([i[0] for i in r])
    m_xs = np.array([i[1] for i in r])
'''
result = model.predict([u_xs, m_xs], batch_size=128).clip(1.0, 5.0)
#result = [np.argmax(r) for r in result]
#result = [np.sum([idx * v for idx, v in enumerate(r)])+1 for r in result]

w = []
for idx, i in enumerate(result):
    w.append([str(idx+1), str(i[0])])
with open(sys.argv[3], 'w') as f:
    f.write('TestDataID,Rating\n' + '\n'.join([','.join(i) for i in w]))
