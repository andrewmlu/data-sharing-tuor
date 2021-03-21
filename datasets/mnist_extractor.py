import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from struct import *
import numpy as np
from util.data import get_one_hot_from_label_index


def mnist_extract_samples(sample_list, is_train=True, file_path=os.path.dirname(__file__), is_fashion_dataset=False):
    # MNIST fashion dataset is from https://github.com/zalandoresearch/fashion-mnist

    if is_fashion_dataset:
        file_path_extended = file_path + '/mnist_fashion'
    else:
        file_path_extended = file_path + '/mnist_original'

    if is_train:
        fImages = open(file_path_extended + '/train-images.idx3-ubyte', 'rb')
        fLabels = open(file_path_extended + '/train-labels.idx1-ubyte', 'rb')
    else:
        fImages = open(file_path_extended + '/t10k-images.idx3-ubyte', 'rb')
        fLabels = open(file_path_extended + '/t10k-labels.idx1-ubyte', 'rb')

    # read the header information in the images file.
    s1, s2, s3, s4 = fImages.read(4), fImages.read(4), fImages.read(4), fImages.read(4)
    mnIm = unpack('>I', s1)[0]
    numIm = unpack('>I', s2)[0]
    rowsIm = unpack('>I', s3)[0]
    colsIm = unpack('>I', s4)[0]

    # read the header information in the labels file and seek to position
    # in the file for the image we want to start on.
    mnL = unpack('>I', fLabels.read(4))[0]
    numL = unpack('>I', fLabels.read(4))[0]

    data = []
    labels = []

    for sample in sample_list:
        fImages.seek(16 + sample * rowsIm * colsIm)
        fLabels.seek(8 + sample)

        # get the input from the image file
        x = np.array(list(fImages.read(rowsIm * colsIm)))/255.0

        # get the correct label from the labels file.
        label = unpack('>B', fLabels.read(1))[0]
        y = get_one_hot_from_label_index(label)

        data.append(x)
        labels.append(y)

    fImages.close()
    fLabels.close()

    return data, labels


def mnist_extract(start_sample_index, num_samples, is_train=True, file_path=os.path.dirname(__file__), is_fashion_dataset=False):
    sample_list = range(start_sample_index, start_sample_index + num_samples)
    return mnist_extract_samples(sample_list, is_train, file_path, is_fashion_dataset)


def mnist_fashion_extract_samples(sample_list, is_train=True, file_path=os.path.dirname(__file__)):
    return mnist_extract_samples(sample_list, is_train, file_path, is_fashion_dataset=True)


def mnist_fashion_extract(start_sample_index, num_samples, is_train=True, file_path=os.path.dirname(__file__)):
    sample_list = range(start_sample_index, start_sample_index + num_samples)
    return mnist_fashion_extract_samples(sample_list, is_train, file_path)