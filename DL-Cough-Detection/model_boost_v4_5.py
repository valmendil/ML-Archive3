#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope


"""
	larger conv net - Train top + bottom + keep the rest at random 
"""


def classify(inputs, 
	     num_classes,
             dropout_keep_prob=0.5,
             middle_size=1,
             bottom_size=1,
             weight_decay = 1e-5,
             fc_size=16,
             num_filter=16,
	     scope=None,
	     reuse=None,
             is_training=True 
	):
        """
	 model used to make predictions
	 input: x -> shape=[None,bands,frames,num_channels]
	 output: logits -> shape=[None,num_labels]
        """
        with slim.arg_scope(simple_arg_scope(weight_decay=weight_decay)): 
        	#with slim.arg_scope(batchnorm_arg_scope()): 
                      with slim.arg_scope([slim.batch_norm, slim.dropout],
		                	is_training=is_training):
                             with tf.variable_scope(scope, 'model_v1', [inputs], reuse=reuse) as scope:

                                      net = tf.expand_dims(inputs, -1) #input needs to be in the format NHWC!! if there is only one channel, expand it by 1 dimension

                                      with tf.variable_scope('bottom'):
                                                #net = slim.conv2d(net, num_filter, [4, 4], rate=2, scope='convB1')
                                                net = slim.repeat(net, bottom_size, slim.conv2d, num_filter, [3, 7], scope='convB2')
                                                net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='poolB1')

				      #random block
                                      with tf.variable_scope('middle'):
                                                net = slim.repeat(net, middle_size, slim.conv2d, num_filter, [3, 5], scope='convM') #, reuse=i>0
                                                net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='poolM')

				      # Use conv2d instead of fully_connected layers.
                                      with tf.variable_scope('top'):
                                                net = slim.flatten(net)
                                                net = slim.fully_connected(net, fc_size, scope='fc1')
                                                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')
                                                logits = slim.fully_connected(net, num_classes, scope='fc2', activation_fn=None) 

                                      return logits




def loss_fkt(logits, y):
        return tf.reduce_mean(tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = y, label_smoothing=0.05)) 



def build_model(x, 
		y,
	        num_classes=2,
		num_estimator=64,#32
                is_training=True,
		reuse=None
		):
        """
	 handle model. calculate the loss and the prediction for some input x and the corresponding labels y
	 input: x shape=[None,bands,frames,num_channels], y shape=[None]
	 output: loss shape=(1), prediction shape=[None]

	CAUTION! controller.py uses a function whith this name and arguments.

	here we do boosting without additive training

        """
        #preprocess
        y = slim.one_hot_encoding(y, num_classes)
        
        #model	
        predictions = classify(x, num_classes=num_classes, is_training=is_training, reuse=reuse, scope='H0')
        logits = predictions 
        loss = loss_fkt(logits, y)

        for i in range(1,num_estimator):
                #logits = tf.stop_gradient(logits)
                predictions = classify(x, num_classes=num_classes, is_training=is_training, reuse=reuse, scope='H%d'%(i+1))
                logits = logits + predictions
                loss += loss_fkt(logits, y)

   
        #results
        predictions = tf.argmax(slim.softmax(logits),1)
        return loss, predictions 	




#Parameters
TRAINABLE_SCOPES = ['top'] #only top is trainable



