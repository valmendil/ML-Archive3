#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope

"""
	densenet like in [https://arxiv.org/pdf/1608.06993.pdf] - train all
"""


@slim.add_arg_scope
def _conv(inputs, num_filters, kernel_size, stride=1, dropout_rate=None,
          scope=None, outputs_collections=None):
  with tf.variable_scope(scope, 'd_conv', [inputs]) as sc:
    net = slim.batch_norm(inputs)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, num_filters, kernel_size)

    if dropout_rate:
      net = slim.dropout(net, dropout_rate)


  return net


@slim.add_arg_scope
def _conv_block(inputs, num_filters, scope=None, outputs_collections=None):
  with tf.variable_scope(scope, 'conv_block', [inputs]) as sc:
    net = inputs
    net = _conv(net, num_filters*4, 1, scope='x1')
    net = _conv(net, num_filters, 3, scope='x2')
    net = tf.concat([inputs, net], axis=3)

  return net


@slim.add_arg_scope
def dense_block(inputs, num_layers, num_filters, growth_rate,
                 grow_num_filters=True, scope=None, outputs_collections=None):

  with tf.variable_scope(scope, 'dense_block', [inputs]) as sc:
    net = inputs
    for i in range(num_layers):
      net = _conv_block(net, growth_rate, scope='conv_block-%d'%(i + 1))

      if grow_num_filters:
        num_filters += growth_rate

  return net, num_filters


@slim.add_arg_scope
def transition_block(inputs, num_filters, compression=1.0,
                      scope=None, outputs_collections=None):

  num_filters = int(num_filters * compression)
  with tf.variable_scope(scope, 'transition_block', [inputs]) as sc:
    net = inputs
    net = _conv(net, num_filters, 1, scope='blk')

    net = slim.max_pool2d(net, [1,2])

  return net, num_filters


def densenet(inputs,
             num_classes, 
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,6,24],#,16],
                  dropout_rate=0.2,
                  is_training=True,
                  reuse=None,
                  scope='densenet121'):

  compression = 1.0 - reduction
  num_dense_blocks = len(num_layers)

  with tf.variable_scope(scope, [inputs, num_classes],
                         reuse=reuse) as sc:
    with slim.arg_scope([_conv], dropout_rate=dropout_rate):
      # initial convolution
      with tf.variable_scope('bottom', [inputs]):
        net = slim.conv2d(inputs, num_filters, [3,7], scope='conv1')

      # blocks
      for i in range(num_dense_blocks - 1):
        # dense blocks
        net, num_filters = dense_block(net, num_layers[i], num_filters,
                                        growth_rate,
                                        scope='dense_block-%d'%(i+1))

        # Add transition_block
        net, num_filters = transition_block(net, num_filters,
                                             compression=compression,
                                             scope='transition_block-%d'%(i+1))


      # final blocks
      with tf.variable_scope('top', [inputs]):
        net, num_filters = dense_block(
              net, num_layers[-1], num_filters,
              growth_rate,
              scope='dense_block-%d'%(num_dense_blocks))

        net = slim.batch_norm(net)
        net = tf.nn.relu(net)
        net = tf.reduce_mean(net, [1,2], name='global_avg_pool')

        net = slim.fully_connected(net, num_classes, activation_fn=None, biases_initializer=tf.zeros_initializer(), scope='logits')

      return net


def densenet_arg_scope(is_training,
                       seed=12,
                       weight_decay=1e-4,
                       batch_norm_decay=0.99,
                       batch_norm_epsilon=1.1e-5):
  with slim.arg_scope([slim.conv2d],
                       weights_regularizer=slim.l2_regularizer(weight_decay),
                       weights_initializer= slim.variance_scaling_initializer(seed=seed),
                       activation_fn=None,
                       biases_initializer=None):
    with slim.arg_scope([slim.batch_norm],
                        scale=True,
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon,
                        updates_collections=tf.GraphKeys.UPDATE_OPS) as scope:
      with slim.arg_scope([slim.batch_norm, slim.dropout],
		                	is_training=is_training):
        return scope



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
        with slim.arg_scope(densenet_arg_scope(is_training)): 
             x = tf.expand_dims(x, -1) 
             logits = densenet(x, num_classes, reuse=reuse)

	#results
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = y)) 
        predictions = tf.argmax(slim.softmax(logits),1)

        return loss, predictions 	




#Parameters
TRAINABLE_SCOPES = None #all weights are trainable



