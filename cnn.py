"""Builds a Convolutional Neural Network for classification of CIFAR-10.

Summary of the class:

#Conv_NNet(conv_layers, fc_layers, conv_kernel, conv_stride, pool_kernel, pool_stride, image_shape, keep_probability,
          num_of_outputs, beta, distortion): Create a Convolutional Neural Network with arbitrary characteristics.

"""
import os
import random
import tensorflow as tf
import numpy as np

import preprocess


class Conv_NNet(object):
    """Create a Convolutional Neural Network with arbitrary characteristics.

    Data augmentation is included in this structure, to add your arbitrary manipulation (such as random ,
    adding noise, random flip). You can add them in distorted_dataset function.

    L_2 regularization is considered in this CNN. Dropout is also included.


    Methods:

    fit(): Create the whole graph for training. \n
    train(train_images, train_labels, valid_images, valid_labels, batch_size, epochs): Train the constructed network. \n
    predict(images, labels, label_names): Predict labels for the given images and find the accuracy based on true labels.
    """

    def __init__(self, conv_layers=[40,80,80], fc_layers=[500,250,125], conv_kernel=(4,4), conv_stride=(1,1),
                 pool_kernel=(2,2), pool_stride=(2,2), image_shape=[32,32,3], keep_probability=.5, num_of_outputs=10,
                 beta=0.001, distortion=True):
        """
        Args:
            conv_layers:  a list of ints with the number of filters used for the convolutional layers
            fc_layers: a list of ints with the number of nodes used for the fully-connected layers
            conv_kernel: a tuple with the filter size of convolutional layers for the height and width of input image
            conv_stride: a tuple with the stride of the sliding convolutional window for the height and width of input
             image
            pool_kernel: a tuple with the size of the pooling window for the height and width of input image
            pool_stride: a tuple with the stride of the sliding pooling window for the height and width of input image
            image_shape: a list of [image_height,image_width,3] with the shape of input image
            keep_probability: a scalar of float with the keeping probability of dropout.
            num_of_outputs: a scalar of int with the number of classes in the classification task
            beta: a scalar of float with the coefficient of L2 regularization
            distortion: bool; True means that the input data distortion is used for the training
             task
        """
        self.keep_probability = keep_probability
        self.distortion = distortion
        self.image_shape = image_shape
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.num_of_outputs = num_of_outputs
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.beta = beta
        # placeholder for the input images
        self.x = tf.placeholder(shape=[None, self.image_shape[0], self.image_shape[1],
                                       self.image_shape[2]], dtype=tf.float32)
        # placeholder for the input image labels
        self.y = tf.placeholder(shape=[None, num_of_outputs], dtype=tf.float32)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        # The following palceholders are defined to be able to switch between
        #  training data and validation data
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.distort = tf.placeholder(dtype=tf.bool)
        self.batch = tf.placeholder(dtype=tf.int64)

    @staticmethod
    def conv_layer(inp, num_of_filters, conv_kernel, conv_stride, pool_kernel, pool_stride):
        """ Create a convolutiona layer followed by a max-pooling function.

        Args:
            inp: a 4-D tensor of size [batch_size, image_height, image_width, 3] with the input images
            num_of_filters: num of layer output filters
            conv_kernel: a tuple with the filter size of convolutional layers for the height and width of input image
            conv_stride: a tuple with the stride of the sliding convolutional window for the height and width of input
             image
            pool_kernel: a tuple with the size of the pooling window for the height and width of input image
            pool_stride: a tuple with the stride of the sliding pooling window for the height and width of input image

        Returns:
            output of the convolutional layer for the given input images
        """
        in_shape = inp.get_shape().as_list()
        stddev = 1/np.sqrt(conv_kernel[0]*conv_kernel[1]*in_shape[3])
        weight = tf.get_variable("weight",initializer=tf.random_normal([conv_kernel[0], conv_kernel[1],
                                                             in_shape[3], num_of_filters], stddev=stddev))
        tf.add_to_collection("L2_Weights", weight)
        bias = tf.get_variable("bias", initializer=tf.zeros(shape=[num_of_filters]))
        # bias = tf.zeros(shape=[num_of_filters])
        conv_val = tf.nn.conv2d(inp, weight, strides=[1, conv_stride[0],
                                                      conv_stride[1], 1], padding='SAME')
        conv_val = tf.nn.bias_add(conv_val, bias)
        conv_val = tf.nn.relu(conv_val)
        output = tf.nn.max_pool(conv_val, [1, pool_kernel[0], pool_kernel[1], 1], [1, pool_stride[0], pool_stride[1], 1],
                                padding='SAME')
        return output

    @staticmethod
    def fc_layer(inp, num_of_outputs):
        """Create a fully_connected layer with Relu activation function.

        Args:
            inp: a 1-D tensor
            num_of_outputs: a scalar of int with the number of nodes in the fc_layer

        Returns:
            output of the fullyconnected layer
        """
        inp_shape = inp.get_shape().as_list()
        weight = tf.get_variable("weight", initializer=tf.random_normal([inp_shape[1], num_of_outputs], stddev=1/np.sqrt(inp_shape[1])))
        tf.add_to_collection("L2_Weights", weight)
        bias = tf.get_variable("bias", initializer=tf.zeros(shape=[num_of_outputs]))
        # bias = tf.zeros(shape=[num_of_outputs])
        fc_layer_val = tf.add(tf.matmul(inp, weight), bias)
        fc_layer_val = tf.nn.relu(fc_layer_val)
        return fc_layer_val

    @staticmethod
    def output_layer(inp, num_of_outputs):
        """Create the output layer for the network.

        Args:
            inp: a 1-D tensor
            num_of_outputs: a scalar of int with the number of classes in the classification task

        Returns:
            output of the network
        """
        in_shape = inp.get_shape().as_list()
        weight = tf.get_variable("weight", initializer=tf.random_normal([in_shape[1], num_of_outputs], stddev=1/np.sqrt(in_shape[1])))
        tf.add_to_collection("L2_Weights", weight)
        bias = tf.get_variable("bias", initializer=tf.zeros(shape=[num_of_outputs]))
        # bias = tf.zeros(shape=[num_of_outputs])
        out_val = tf.add(tf.matmul(inp, weight), bias)
        return out_val

    def forward(self,inp_batch, keep_pro):
        """Create the network's graph from the first convolutional layer to the output layer.

        Args:
            inp_batch: a 4-D tensor of size [batch_size, image_height, image_width, 3] with the
             input images
            keep_pro: a scalar of float with the keeping probability of dropout.

        Returns:
            output of the network
        """
        for i in range(len(self.conv_layers)):
            if i == 0:
                with tf.variable_scope('conv_layer'+str(i), reuse=tf.AUTO_REUSE):
                    layer = self.conv_layer(inp=inp_batch, num_of_filters=self.conv_layers[i], conv_kernel=self.conv_kernel,
                                            conv_stride=self.conv_stride, pool_kernel=self.pool_kernel,
                                            pool_stride=self.pool_stride)
            else:
                with tf.variable_scope('conv_layer'+str(i), reuse=tf.AUTO_REUSE):
                    layer = self.conv_layer(inp=layer, num_of_filters=self.conv_layers[i], conv_kernel=self.conv_kernel,
                                            conv_stride=self.conv_stride, pool_kernel=self.pool_kernel,
                                            pool_stride=self.pool_stride)

        if len(self.conv_layers) == 0:
            layer = inp_batch
        else:
            layer_shape = layer.get_shape().as_list()
            layer_length = layer_shape[1] * layer_shape[2] * layer_shape[3]
            layer = tf.reshape(layer, [-1, layer_length])

        for j in range(len(self.fc_layers)):
            with tf.variable_scope('fc_layer'+str(j), reuse=tf.AUTO_REUSE):
                layer = self.fc_layer(inp=layer, num_of_outputs=self.fc_layers[j])
                layer = tf.nn.dropout(layer, keep_prob=keep_pro)
        with tf.variable_scope('output_layer', reuse=tf.AUTO_REUSE):
            output = self.output_layer(inp=layer, num_of_outputs=self.num_of_outputs)
        return output

    @staticmethod
    def normalization(image, label):
        """Normalize the image to 1.

        Args:
            image: A 3-D tensor of size [batch_size, image_height, image_width, 3]
            label: A scalar showing the label corresponding to the input image

        Returns:
            Normalized image, label
        """
        return tf.divide(image, float(256)), label

    @staticmethod
    def random_rotation(image, label):
        """Rotate the image randomly.

       Args:
           image: A 3-D tensor of size [batch_size, image_height, image_width, 3]
           label: A scalar showing the label corresponding to the input image

       Returns:
           rotated image, label
       """
        return tf.contrib.image.rotate(image, random.randrange(-10, 10)), label

    @staticmethod
    def random_flip(image, label):
        """Flip the image randomly.
       Args:
           image: A 3-D tensor of size [batch_size, image_height, image_width, 3]
           label: A scalar showing the label corresponding to the input image

       Returns:
           flipped image, label
        """
        return tf.image.random_flip_left_right(image), label

    def distorted_dataset(self):
        """Incorporate the defined distortions into the dataset."""

        # add the required distortion here

        self.dataset = self.dataset.map(self.normalization)
        #self.dataset = self.dataset.map(self.random_rotation)
        self.dataset = self.dataset.map(self.random_flip)

    def fit(self):
        """Create the whole graph for training."""

        if self.distort == True:
            self.distorted_dataset()
        self.dataset = self.dataset.batch(self.batch)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_image_batch, self.next_label_batch = self.iterator.get_next()
        self.output = self.forward(self.next_image_batch, self.keep_prob)

        #Compute sum of l2 norm of the weights
        l2 = tf.zeros([],dtype=tf.float32);
        for k in range(len(tf.get_collection("L2_Weights"))):
            l2 = tf.add(l2,tf.nn.l2_loss(tf.get_collection("L2_Weights")[k]))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.next_label_batch))
        self.cost = tf.reduce_mean(loss+self.beta*l2)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

    @staticmethod
    def accuracy(output, labels):
        """Calculate the accuracy of output according to the labels.

        Args:
           output: A 2-D tensor of size [num_of_images, num_of_classes], coming as predicted output of the network.
           labels: A 2-D tensor of size [num_of_images, num_of_classes], coming as true labels corresponding to the
            input images of the network.

        Returns:
           accuracy of the results
        """
        correct_prediction = tf.equal(tf.argmax(labels, axis=1), tf.argmax(output, axis=1))
        return tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    def train(self, train_data, train_labels, valid_images, valid_labels, batch_size, epochs):
        """Train the network.

        Args:
           train_data: A 4-D array of size [num_of_training_images, image_height, image_width, 3], containing the
            training images
           train_labels: A 2-D array of size [num_of_training_images, num_of_classes], containing the (hot-encoded)
            training image labels
           valid_images: A 4-D array of size [num_of_validation_images, image_height, image_width, 3], containing the
            validation images
           valid_labels: A 2-D array of size [num_of_training_images, num_of_classes], containing the (hot-encoded)
            validation image labels
           batch_size: A scalar showing the batch_size
           epochs: A scalar showing the number of epochs
        """
        saver = tf.train.Saver()
        save_dir = "."
        ck_dir = os.path.join(save_dir, 'checkpoints')
        if not os.path.isdir(ck_dir):
            os.mkdir(ck_dir)
        save_path = ck_dir + '/cifar10_variables'
        with tf.Session() as sess:
            try:
                print("Restoring the weights from the latest checkpoint ...")
                last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir = ck_dir)
                saver.restore(sess, save_path=last_checkpoint)
                print("Weights restored successfully! ")
            except:
                print("No checkpoint found!")
                sess.run(tf.global_variables_initializer())
                print("Network weights initialized.")
            # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            # print(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
            # print(tf.get_collection("L2_Weights"))
            print("Training...")
            for epoch in range(epochs):
                if epoch%2 == 0:
                    saver.save(sess, save_path= save_path)
                sess.run(self.iterator.initializer, feed_dict={self.x: train_data,
                                                               self.y: train_labels,
                                                               self.batch: batch_size})
                while True:
                    try:
                        feed_dict = {self.keep_prob: .75, self.distort : self.distortion}
                        _, cst, train_accuracy = sess.run([self.optimizer, self.cost,
                                                          self.accuracy(self.output, self.next_label_batch)],
                                                          feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        break

                sess.run(self.iterator.initializer, feed_dict={self.x: valid_images,
                                                               self.y: valid_labels,
                                                               self.batch: len(valid_images)})
                valid_accuracy = sess.run(self.accuracy(self.output, self.next_label_batch),
                                          {self.keep_prob: 1, self.distort: False})
                print('Epoch:{},  Cost: {:.5f}, Train accuracy: {:.5f}, Validation accuracy: {:.5f}'
                      .format(epoch + 1, cst, train_accuracy, valid_accuracy))

    def test(self, images, labels, label_names):
        """Predict labels for the given images and find the accuracy based on the true labels.

        Args:
           images: A 4-D array of size [num_of_images, image_height, image_width, 3], containing the
            images
           labels: A 2-D array of size [num_of_training_images, num_of_classes], containing the (hot-encoded)
            image true labels
           label_names: A list containing the name of classes
        """
        saver = tf.train.Saver()
        save_dir = "."
        ck_dir = os.path.join(save_dir, 'checkpoints')
        save_path = ck_dir + '/cifar10_variables'
        with tf.Session() as sess:
            saver.restore(sess, save_path=save_path)
            sess.run(self.iterator.initializer, feed_dict={self.x: images,
                                                       self.y: labels,
                                                       self.batch: len(images)})
            predicted_outputs = sess.run(tf.argmax(self.output, axis=1), {self.keep_prob: 1, self.distort: False})
        correct_prediction = np.equal(predicted_outputs, np.argmax(labels, axis=1))
        false_predictions = [i for i, x in enumerate(correct_prediction) if not x]
        print("False predicted image indices: {}".format(false_predictions))
        print("Test Accuracy: {:.3f}".format(np.mean(correct_prediction)))
        # lnames_array = np.array(label_names)


