from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import csv
import numpy as np
import codecs
import pickle

model = load_model(sys.argv[1])

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with codecs.open(sys.argv[2], 'r', encoding='utf-8') as file:
    xs = tokenizer.texts_to_sequences(file.readlines())
    xs = tokenizer.sequences_to_matrix(xs)

result = model.predict(xs, batch_size=128)
print(result)
result = np.around(result)

w = []
for idx, i in enumerate(result):
    w.append([str(idx), str(int(round(i[0])))])
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n' + '\n'.join([','.join(i) for i in w]))