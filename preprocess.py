"""Generates the CIFAR10 train, validation and test data for the network.

Summary of functions:
#unpickle(file)
#hot_encode(labels)
#normalize(x): Normalize the image x by dividing it by 256
#data_pickle_generation(path): Process the downloaded CIFAR-10 data at the given path and re-save them in proper format
#read_train_data(): Return the train and validation images and labels in numpy arrays, and also the list of label names.
#read_test_data(): Return the test images and labels in numpy arrays
"""
import os
import random
from random import shuffle
import numpy as np
import pickle
import tensorflow as tf
# from matplotlib import pyplot as plt
# plt.ion()
# import patoolib

#if not os.path.isdir('C:\Users\shmou\PycharmProjects\CNN\cifar-10-batches-py'):
#    patoolib._extract_archive('cifar-10-python.tar.gz',outdir='.')

# Directory of the raw data pickle files
raw_data_dir = "."


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def hot_encode(labels):
    output = (np.array(labels)[:, None] == np.array(range(0, 10))).astype(np.int32)
    return output


def normalize(x):
    return x/float(256)


def data_pickle_generation(path):
    """Process the downloaded and extracted CIFAR-10 data and re-save them.

    Arg:
    path: the path of the downloaded and extracted data
    """
    preprocessed_file_dir = os.path.join(path, 'preprocessed files')
    if not os.path.isdir(preprocessed_file_dir):
        os.mkdir(path + '\preprocessed files')
    preprocessed_train_file = preprocessed_file_dir + '/train_data'
    preprocessed_valid_file = preprocessed_file_dir + '/valid_data'
    if not os.path.isfile(preprocessed_train_file):
        # os.mkdir(path+'\preprocessed files')
        train_images = np.empty((0,32,32,3))
        train_labels = np.empty((0,10))
        valid_images = np.empty((0,32,32,3))
        valid_labels = np.empty((0,10))
        for batch_num in range(1,6):
            raw_data_path = os.path.join(path, 'data_batch_' + str(batch_num))
            try:
                raw_data = unpickle(raw_data_path)
                print('Processing data_batch_{} ...'.format(batch_num))
                raw_labels = raw_data[b'labels']
                raw_labels = hot_encode(raw_labels)
                raw_images = raw_data[b'data']
                raw_images = np.reshape(raw_images, [-1, 3, 32, 32]).transpose(0, 2, 3, 1)
                assert len(raw_labels) == len(raw_images)
                num_of_data = len(raw_labels)
                a = np.arange(num_of_data)
                shuffle(a)
                raw_images = raw_images[a]
                raw_labels = raw_labels[a]
                valid_images = np.append(valid_images, raw_images[:int(.1 * num_of_data)], axis=0)
                valid_labels = np.append(valid_labels, raw_labels[:int(.1 * num_of_data)], axis=0)
                train_images = np.append(train_images, raw_images[int(.1 * num_of_data):], axis=0)
                train_labels = np.append(train_labels, raw_labels[int(.1 * num_of_data):], axis=0)
            except FileNotFoundError:
                print('File data_batch_{} does not exist!'.format(batch_num))
        assert len(train_images) == len(train_labels)
        assert len(valid_images) == len(valid_labels)
        print('Train Images Dataset Shape:', np.shape(train_images))
        print('Train Labels Dataset Shape:', np.shape(train_labels))
        print('Validation Images Dataset Shape:', np.shape(valid_images))
        print('Validation Labels Dataset Shape:', np.shape(valid_labels))
        assert len(train_images) > 0
        pickle.dump((train_images, train_labels), open(preprocessed_train_file, 'wb'))
        pickle.dump((valid_images, valid_labels), open(preprocessed_valid_file, 'wb'))

        preprocessed_test_file = preprocessed_file_dir + '/test_data'
        raw_test_data_path = os.path.join(path, 'test_batch')
        try:
            raw_test_data = unpickle(raw_test_data_path)
            print('Processing test_batch ...')
            raw_test_labels = raw_test_data[b'labels']
            test_labels = hot_encode(raw_test_labels)
            raw_test_images = raw_test_data[b'data']
            test_images = np.reshape(raw_test_images, [-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            assert len(test_labels) == len(test_images)
        except FileNotFoundError:
            print('File test_batch does not exist!')
        assert len(test_images) == len(test_labels)
        print('Test Images Dataset Shape:', np.shape(test_images))
        print('Test Labels Dataset Shape:', np.shape(test_labels))
        pickle.dump((test_images, test_labels), open(preprocessed_test_file, 'wb'))


def read_train_data():
    """Read the processed training data.

    Returns:
    train_images: A 4-D array of size [num_of_training_images, image_height, image_width, 3], containing the
     training images
    train_labels: A 2-D array of size [num_of_training_images, num_of_classes], containing the (hot-encoded)
     training image labels
    valid_images: A 4-D array of size [num_of_validation_images, image_height, image_width, 3], containing the
     validation images
    valid_labels: A 2-D array of size [num_of_training_images, num_of_classes], containing the (hot-encoded)
     validation image labels
    label_names: A list containing the name of classes
    """
    preprocessed_file_dir = os.path.join(raw_data_dir, 'preprocessed files')
    data_pickle_generation(raw_data_dir)
    train_images, train_labels = pickle.load(open(preprocessed_file_dir+'/train_data', mode='rb'))
    valid_images, valid_labels = pickle.load(open(preprocessed_file_dir+'/valid_data', mode='rb'))
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return train_images, train_labels, valid_images, valid_labels, label_names


def read_test_data():
    """Read the processed test data.

    Returns:
    test_images: A 4-D array of size [num_of_test_images, image_height, image_width, 3], containing the
     test images
    test_labels: A 2-D array of size [num_of_test_images, num_of_classes], containing the (hot-encoded)
     test image labels
    """
    preprocessed_file_dir = os.path.join(raw_data_dir, 'preprocessed files')
    test_images, test_labels = pickle.load(open(preprocessed_file_dir+'/test_data', mode='rb'))
    return test_images, test_labels
