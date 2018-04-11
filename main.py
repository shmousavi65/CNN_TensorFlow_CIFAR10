""" The main file which must be executed!

Note: The CIFAR10 files should be extracted and located in the same directory as this code file
"""

import numpy as np
import random
import tensorflow as tf
import os

# from matplotlib import pyplot as plt
# plt.ion()

from cnn import Conv_NNet
import preprocess


# Read the train and validation data in numpy format
train_images, train_labels, valid_images, valid_labels, label_names = preprocess.read_train_data()

# Read the test and validation data in numpy format
test_images, test_labels = preprocess.read_test_data()

tf.reset_default_graph()

# Create a Convolutional Neural Network
my_Net = Conv_NNet()
my_Net.fit()

# Train the Network
my_Net.train(train_data=train_images, train_labels=train_labels, valid_images=valid_images,
             valid_labels=valid_labels, batch_size=128, epochs=200)

#Test the network
# my_Net.test(test_images[1:10000], test_labels[1:10000], label_names)
