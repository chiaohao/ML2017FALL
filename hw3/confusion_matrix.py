from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

with open(sys.argv[2]) as csvfile:
    r = list(csv.reader(csvfile))
    r = r[1:]
    ys = np_utils.to_categorical([int(i[0]) for i in r])
    xs = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs = xs.reshape(-1,48,48,1) / 255.0
    d = int(xs.shape[0] * 0.2)
    xs_train = xs[:-d]
    ys_train = ys[:-d]

model = load_model(sys.argv[1])
np.set_printoptions(precision=2)
p = model.predict(xs_train)
p = [r.argmax() for r in p]
a = [r.argmax() for r in ys_train]
cm = confusion_matrix(a, p)

fig = plt.figure()
plot_confusion_matrix(cm, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()
fig.savefig('confusion.png')
