#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope

"""
	resnet 18 like in [https://arxiv.org/pdf/1512.03385.pdf] - train all
"""

def classify(x, 
	     num_classes,
             num_layers=[2,2,2],#,2],
	     scope='resnet_18_v1',
	     reuse=None,
             is_training=True
	):
	"""
	 resnet-18 used to make predictions
	"""
	with slim.arg_scope(simple_arg_scope()): 
		with slim.arg_scope(batchnorm_arg_scope()): 
			with slim.arg_scope([slim.batch_norm, slim.dropout],
		                	is_training=is_training):
  				with slim.arg_scope([slim.conv2d], weights_initializer= slim.variance_scaling_initializer(seed=0)):
		                     with tf.variable_scope(scope, [x], reuse=reuse) as sc:

                                                x = tf.expand_dims(x, -1) #input needs to be in the format NHWC!! if there is only one channel, expand it by 1 dimension
                                                num_blocks = len(num_layers)
                                                stride=1

                                                # initial convolution
                                                with tf.variable_scope('bottom', [x]):
                                                     shortcut = x = slim.conv2d(x, 64, [3,7], scope='convInit')
                                                     #shortcut = x = slim.avg_pool2d(x, [1,2], 2)

                                                #resnet blocks
                                                for i in range(num_blocks):
                                                         with tf.variable_scope("block%d"%i):
                                                                depth = 2**(i+6) #64,128,256,512

                                                                for j in range(num_layers[i]):
                                                                      residual = slim.conv2d(x, depth, [3,3], stride, scope='convA%d.%d'%(i,j))
                                                                      residual = slim.conv2d(residual, depth, [3,3], activation_fn=None, scope='convB%d.%d'%(i,j))
                                                                      shortcut = x = tf.nn.relu(shortcut + residual)
                                                                      stride=1

                                                                if i+1 < num_blocks:
                                                                      shortcut = slim.conv2d(x,  2**(i+6+1), 1, stride=2, scope='convS.%d'%i)
                                                                      stride=2

                                                #print (x.get_shape())

                                                # initial convolution
                                                with tf.variable_scope('top', [x]):
                                                     x = tf.reduce_mean(x, [1, 2], name='global_pool')
                                                     logits = slim.fully_connected(x, num_classes, scope='logits', activation_fn=None)

                                                return logits


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
        logits = classify(x, num_classes=num_classes, is_training=is_training, reuse=reuse)	

	#results
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = y)) 
        predictions = tf.argmax(slim.softmax(logits),1)

        return loss, predictions 	




#Parameters
TRAINABLE_SCOPES = None #all weights are trainable



