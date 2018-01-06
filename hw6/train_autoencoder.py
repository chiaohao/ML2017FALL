import numpy as np
import sys
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

data = np.load(sys.argv[1])
data = data.reshape((-1,28,28,1)) / 255.

d = int(len(data) * 0.05)
x_train = data[0:-d]
x_val = data[-d:]

input_image = Input(shape=(28,28,1))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_image)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(256, activation='sigmoid')(x)
x = Dense(128, activation='sigmoid')(x)
encoded = Dense(64, activation='sigmoid')(x)

x = Dense(128, activation='sigmoid')(encoded)
x = Dense(256, activation='sigmoid')(x)
x = Dense(128, activation='sigmoid')(x)
x = Reshape((4, 4, 8))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_image, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

encoder = Model(input_image, encoded)

class ModelCheckpointWithEncoder(ModelCheckpoint):
    global encoder
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                            encoder.save_weights('encoder_' + filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                            encoder.save('encoder_' + filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                    encoder.save_weights('encoder_' + filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                    encoder.save('encoder_' + filepath, overwrite=True)

mc = ModelCheckpointWithEncoder(filepath='model_128.h5', save_best_only=True)
e = EarlyStopping(patience=5)

autoencoder.fit(x_train, x_train, 
        epochs=100, 
        batch_size=128, 
        validation_data=(x_val, x_val),
        callbacks=[mc, e])

#p = encoder.predict(x_train, batch_size=128)
#print(p.shape)
