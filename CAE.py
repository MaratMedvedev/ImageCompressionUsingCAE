import time
import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np


def build_cae(img_shape, code_size):
    # encoder
    encoder = tf.keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))

    encoder.add(L.Conv2D(filters=32, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))

    # decoder
    decoder = tf.keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))

    decoder.add(L.Dense((img_shape[0] // 2) * (img_shape[1] // 2)))
    decoder.add(L.Reshape((img_shape[0] // 2, img_shape[1] // 2, 1)))
    decoder.add(L.Conv2DTranspose(filters=300, kernel_size=(3, 3), strides=1, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=200, kernel_size=(3, 3), strides=1, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=100, kernel_size=(3, 3), strides=1, activation='elu', padding='same'))
    decoder.add(
        L.Conv2DTranspose(filters=img_shape[2], kernel_size=(3, 3), strides=2, activation=None,
                          padding='same'))
    return encoder, decoder


class EarlyStoppingAtMinLossOrMaxTime(tf.keras.callbacks.Callback):
    def __init__(self, patience: int = 0, max_t: int = 10**9, max_epoch=-1):
        super().__init__()
        self.patience = patience
        self.max_t = max_t
        self.max_epoch = max_epoch

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.Inf
        self.st_t = time.time()
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        past = int(time.time() - self.st_t)
        predicted = (past / (epoch+1)) * self.max_epoch
        t = min(int(predicted), self.max_t - past)
        print(f"\tThere are approximately {t // 3600}h {(t % 3600) // 60}m {(t % 3600) % 60}s left to complete")
        if 0 < self.max_t <= time.time() - self.st_t:
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)

        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
