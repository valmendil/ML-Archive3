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
	num_cells=1,
	scope=None,
	reuse=None
	):
  with tf.variable_scope('rnn_multicell', [inputs],
      			reuse=reuse) as sc:
            cell1 = tf.nn.rnn_cell.GRUCell(128)
            cell2 = tf.nn.rnn_cell.GRUCell(64)
            cell3 = tf.nn.rnn_cell.GRUCell(32)
            cell4 = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple = True)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3, cell4])
            output, state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
            output = tf.transpose(output, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)
            last = slim.fully_connected(last,128, 
					activation_fn=tf.nn.relu)
            return slim.fully_connected(last,2, 
                                        activation_fn=None)


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




