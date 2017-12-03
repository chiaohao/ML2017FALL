from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import os
import csv
import numpy as np
import codecs
import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with codecs.open(sys.argv[1], 'r', encoding='utf-8') as file:
    r = [i.split(',', 1)[1] for i in file.readlines()][1:]
    xs = pad_sequences(tokenizer.texts_to_sequences(r), maxlen=37)

result_sum = []
dir_list = os.listdir(sys.argv[3])
for i in range(len(dir_list)):
    print('loading model : %s' % dir_list[i])
    model = load_model(sys.argv[3] + '/' + dir_list[i])
    
    result = model.predict(xs, batch_size=128)
    result = np.around(result)
    if result_sum == []:
        result_sum = result
    else:
        result_sum += result

result_sum = np.around(result_sum / len(dir_list))

w = []
for idx, i in enumerate(result_sum):
    w.append([str(idx), str(int(round(i[0])))])
with open(sys.argv[2], 'w') as f:
    f.write('id,label\n' + '\n'.join([','.join(i) for i in w]))
