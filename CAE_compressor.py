import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np
import matplotlib.pyplot as plt
from image_dataset import get_image_dataset
from CAE import build_cae, EarlyStoppingAtMinLossOrMaxTime
import sys
import os
from zipfile import ZipFile
import struct


class CAEcompressor:
    def __init__(self, epoch: int = 30, code_size: int = -1, time: int = 10 ** 9, visualize: bool = False):

        self.decoder = None
        self.encoder = None
        self.img_shape = None
        self.X = None
        self.epoch = epoch
        self.time = time
        self.visualize = visualize
        self.code_size = code_size

    def compress(self, data_path, save_path):
        self.X, self.data_type = get_image_dataset(data_path)
        self.img_shape = self.X[0].shape
        if self.code_size < 1:
            self.code_size = int(((self.img_shape[0] * self.img_shape[1]) ** 0.5) // 2)
        print("Training model...")
        self.encoder, self.decoder = self.build_and_train_cae()
        print("Encoding images...")
        codes = self.encoder.predict(self.X)
        if self.visualize:
            self.visualize_compressing()
        self.decoder.save_weights('decoder.h5')
        with open("codes.f32", "wb") as f:
            for code in codes.flatten():
                b = struct.pack('f', code)
                f.write(b)
        with open("config.txt", "w") as f:
            f.write(str(len(self.X)) + '\n'
                    + 'x'.join([str(i) for i in self.img_shape]) + '\n'
                    + f"{self.code_size}\n"
                      f"{self.data_type}")
        with ZipFile(save_path, 'w') as zipObj2:
            zipObj2.write('config.txt')
            zipObj2.write('codes.f32')
            zipObj2.write('decoder.h5')
            os.remove('config.txt')
            os.remove('codes.f32')
            os.remove('decoder.h5')
        print(f"Your compressed data located in '{save_path}' file")

    def build_and_train_cae(self):
        if self.check_model():
            encoder, decoder = build_cae(self.img_shape, self.code_size)
            return self.train_cae(encoder, decoder)

    def train_cae(self, encoder, decoder):
        inp = L.Input(self.img_shape)
        code = encoder(inp)
        reconstruction = decoder(code)
        autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
        autoencoder.compile(optimizer="adamax", loss='mse')
        autoencoder.fit(x=self.X, y=self.X, epochs=self.epoch,
                        batch_size=int(len(self.X) ** 0.5),
                        callbacks=[EarlyStoppingAtMinLossOrMaxTime(max_t=self.time, max_epoch=self.epoch)]
                        )
        return encoder, decoder

    def visualize_compressing(self):
        def show_image(img):
            plt.imshow(np.rint(img * 255).astype(int))

        def visualize(img, encoder, decoder):
            """Draws original, encoded and decoded images"""
            code = encoder.predict(img[None])[0]  # img[None] is the same as img[np.newaxis, :]
            reco = decoder.predict(code[None])[0]

            plt.subplot(1, 3, 1)
            plt.title("Original")
            show_image(img)

            plt.subplot(1, 3, 3)
            plt.title("Reconstructed")
            show_image(reco)
            plt.show()

        for i in range(0, min(120, len(self.X)), 6):
            img = self.X[i]
            print(i)
            visualize(img, self.encoder, self.decoder)

    def check_model(self):
        # Check autoencoder shapes along different code_sizes
        get_dim = lambda layer: np.prod(layer.output_shape[1:])
        tf.keras.backend.clear_session()
        encoder, decoder = build_cae(self.img_shape, self.code_size)
        assert encoder.output_shape[1:] == (self.code_size,), "encoder must output a code of required size"
        assert decoder.output_shape[1:] == self.img_shape, "decoder must output an image of valid shape"
        assert len(encoder.trainable_weights) >= 6, "encoder must contain at least 3 layers"
        assert len(decoder.trainable_weights) >= 6, "decoder must contain at least 3 layers"

        for layer in encoder.layers + decoder.layers:
            assert get_dim(layer) >= self.code_size, "Encoder layer %s is smaller than bottleneck (%i units)" % (
                layer.name, get_dim(layer))
        return True


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.25
sess = tf.compat.v1.Session(config=config)

data_path = sys.argv[1]
max_time = sys.argv[2] if len(sys.argv) > 2 else 10 ** 9  # Example of input: '5h 20m 5s' or '50m 5s' or '30s'
if max_time != 10 ** 9:
    times = max_time.split()
    max_time = 0
    sec = {'h': 3600, 'm': 60, 's': 1}
    for i in range(len(times)):
        max_time += int(times[i][:-1]) * sec[times[i][-1]]
save_path = f'{data_path[:-data_path[::-1].index(".") - 1]}_compressed.zip'
cmpsr = CAEcompressor(epoch=100, time=max_time)
cmpsr.compress(data_path, save_path)
