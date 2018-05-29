import tensorflow as tf
import numpy as np
class QuestionClassifier(object):
    def __init__(
      self, input_size, n_classes, vocabulary_size):
        self.question = tf.placeholder(tf.int32, [None, input_size])
        self.label = tf.placeholder(tf.float32, [None, n_classes])
        self.dropout_rate = tf.placeholder(tf.float32)

        # Embedding layer
        W = tf.Variable(tf.random_uniform([vocabulary_size, 128], -1.0, 1.0))
        self.embedded_chars = tf.nn.embedding_lookup(W, self.question)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Convolution Layer: filter size 3
        W1 = tf.Variable(tf.truncated_normal([3, 128, 1, 128], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[128]))
        conv1 = tf.nn.conv2d(self.embedded_chars_expanded, W1, strides=[1, 1, 1, 1],padding="VALID")
        pool1 = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(conv1, b1)), ksize=[1, input_size - 3 + 1, 1, 1],
                               strides=[1, 1, 1, 1],padding='VALID')

        # Convolution Layer: filter size 4
        W2 = tf.Variable(tf.truncated_normal([4, 128, 1, 128], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[128]))
        conv2 = tf.nn.conv2d(self.embedded_chars_expanded, W2, strides=[1, 1, 1, 1], padding="VALID")
        pool2 = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(conv2, b2)), ksize=[1, input_size - 4 + 1, 1, 1],
                               strides=[1, 1, 1, 1], padding='VALID')

        # Convolution Layer: filter size 5
        W3 = tf.Variable(tf.truncated_normal([5, 128, 1, 128], stddev=0.1))
        b3 = tf.Variable(tf.constant(0.1, shape=[128]))
        conv3 = tf.nn.conv2d(self.embedded_chars_expanded, W3, strides=[1, 1, 1, 1], padding="VALID")
        pool3 = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(conv3, b3)), ksize=[1, input_size - 5 + 1, 1, 1],
                               strides=[1, 1, 1, 1], padding='VALID')

        # Combine all the output of all convolution layers
        self.flat_pool = tf.reshape(tf.concat([pool1, pool2, pool3], 3), [-1, 128 * 3])
        self.dropout = tf.nn.dropout(self.flat_pool, self.dropout_rate)

        # Output Layer
        self.scores = tf.layers.dense(self.dropout, n_classes, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      bias_initializer= tf.constant_initializer(0.1))
        self.predictions = tf.argmax(self.scores, 1)

        # Loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.label))

        # Accuracy
        correct = tf.equal(self.predictions, tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, "float"))

        # Training
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))

