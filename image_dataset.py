import zipfile
import os
from PIL import Image
import numpy as np
import cv2


def get_image_dataset(path):
    '''
    :param path: Path to zip file with one video or images
    :return: Dataset that can be used by model during training
    '''
    videoFile = None
    if path[-3:] == 'mp4':
        videoFile = path
    else:
        with zipfile.ZipFile(path, 'r') as zip:
            files = zip.namelist()
            for file in files:
                filename, extension = os.path.splitext(file)
                if extension == '.jpg':
                    zip.extract(file, '/images_to_compression/')
                if extension == '.mp4':
                    zip.extract(file, '/video_to_compression/')
                    videoFile = 'video_to_compression/' + file
                    break
    if videoFile:
        return get_dataset_from_video(videoFile), 'video'
    else:
        return get_dataset_from_images(files), 'images_folder'


def get_dataset_from_video(path):
    print("Preparing video for model...")
    vidcap = cv2.VideoCapture(path)

    frames = []
    success, image = vidcap.read()
    while success:
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        success, image = vidcap.read()
    vidcap.release()
    cv2.destroyAllWindows()
    print("Finish video preprocessing")
    data = []
    for frame in frames:
        frame = frame.astype('float32') / 255.
        data.append(frame)
    return np.array(data)


def get_dataset_from_images(files):
    print("Preparing images for model...")
    mxh = 0
    mxw = 0
    for file in files:
        im = np.array(Image.open('/images_to_compression/' + file, 'r'))
        mxh = max(im.shape[0], mxh)
        mxw = max(im.shape[1], mxw)

    res = np.zeros((len(files), mxh, mxw, 3))
    for i in range(len(files)):
        im = np.array(Image.open('/images_to_compression/' + files[i], 'r'))
        if len(im.shape) == 2:  # Means that image is grayscale.
            im = np.pad(im, ((0, mxh - im.shape[0]), (0, mxw - im.shape[1])), 'constant')
            im = np.stack((im,) * 3, axis=-1)  # to RGB
        else:
            background = np.zeros((mxh, mxw, 3))
            for ii in range(im.shape[0]):
                for jj in range(im.shape[1]):
                    background[ii][jj] = im[ii][jj].copy()
            im = background.copy()
        res[i] = im
    print("Finish images preprocessing")
    return res / 255.
