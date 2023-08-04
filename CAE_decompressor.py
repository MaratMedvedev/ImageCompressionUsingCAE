from CAE import build_cae
from PIL import Image
import cv2
import numpy as np
import zipfile
import os
import sys
import struct


def decompress(data_path, save_folder):
    with zipfile.ZipFile(data_path, 'r') as zip:
        files = zip.namelist()
        for file in files:
            zip.extract(file, '/temp_folder/')

    with open('/temp_folder/config.txt', 'r') as f:
        n_img, img_sh, code_size, data_type = f.read().split('\n')
        n_img = int(n_img)
        img_shape = tuple([int(i) for i in img_sh.split('x')])
        code_size = int(code_size)

    FLOAT_SIZE = 4
    NUM_OFFSETS = code_size * n_img
    filename = '/temp_folder/codes.f32'

    offset_values = [i * FLOAT_SIZE for i in range(NUM_OFFSETS)]
    codes = []
    with open(filename, 'rb') as file:
        for offset in offset_values:
            file.seek(offset)
            buf = file.read(FLOAT_SIZE)
            value = struct.unpack('f', buf)[0]
            codes.append(value)
    codes = np.array(codes).reshape(n_img, code_size)
    encoder, decoder = build_cae(img_shape, code_size)
    decoder.load_weights("/temp_folder/decoder.h5")
    array_images = decoder.predict([codes])

    if save_folder != '':
        os.mkdir(save_folder)
        save_folder += '/'

    if data_type == 'video':
        save_video(f'{save_folder}/{data_path.split("/")[-1]}', preproc_imgs(array_images))
    else:  # Images folder
        for i in range(len(array_images)):
            save_arr_as_image(array_images[i], i)


def preproc_imgs(frames):
    imgs = []
    for frame in frames:
        frame[frame > 0.999] = 1.
        img = cv2.cvtColor(np.uint8(np.rint(frame * 255).astype(int)), cv2.COLOR_RGB2BGR)

        imgs.append(img)
    return imgs


def save_arr_as_image(arr, i):
    arr[arr > 1] = 1.
    img = Image.fromarray(np.uint8(np.rint(arr * 255.))).convert('RGB')
    img.save(f"{save_folder}/img_{i}.jpg")


def save_video(filename, frames):
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, frames[0].shape[:2][::-1])

    for frame in frames:
        out.write(frame)
    out.release()


data_path = sys.argv[1]
save_folder = sys.argv[2] if len(sys.argv) > 2 else data_path[:-data_path[::-1].index('.') - 1] + "_decompressed"
decompress(data_path, save_folder)
