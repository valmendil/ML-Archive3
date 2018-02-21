#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope


"""
	RNN
"""


def RNN_multicell(inputs, 
	num_outputs,
	num_hidden=1024,
	num_cells=1,
        activation_fn=None,
	reuse=None
	):
  with tf.variable_scope('rnn_multicell', [inputs],
      			reuse=reuse) as sc:
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple = True) for i in range(num_cells)])
            output, state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
            output = tf.transpose(output, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)
            return slim.fully_connected(last, 2,
					activation_fn=activation_fn)


def build_model(x, 
		y,
	        num_classes=2,
                is_training=True,
		reuse=None
		):
        """
	 handle model. calculate the loss and the prediction for some input x and the corresponding labels y
	 input: x shape=[None,bands,frames,num_channels], y shape=[None]
	 output: loss shape=(1), prediction shape=[None]

	CAUTION! controller.py uses a function whith this name and arguments.
        """
        #preprocess
        y = slim.one_hot_encoding(y, num_classes)

        #model
        logits = RNN_multicell(x, num_outputs=num_classes, reuse=reuse)	

        #results
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = y)) 
        predictions = tf.argmax(slim.softmax(logits),1)

        return loss, predictions 	





#Parameters
TRAINABLE_SCOPES = None #all weights are trainable




