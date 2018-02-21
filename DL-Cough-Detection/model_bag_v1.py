#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope


"""
	ensemble of conv nets
"""


def classify(inputs, 
             num_estimator,
	     num_classes,
             dropout_keep_prob=0.5,
	     scope=None,
	     reuse=None,
             is_training=True 
	):
        """
	 model used to make predictions
	 input: x -> shape=[None,bands,frames,num_channels]
	 output: logits -> shape=[None,num_labels]
        """
        with slim.arg_scope(simple_arg_scope()): 
        	#with slim.arg_scope(batchnorm_arg_scope()): 
                      with slim.arg_scope([slim.batch_norm, slim.dropout],
		                	is_training=is_training):
                             with tf.variable_scope(scope, 'model_v1', [inputs], reuse=reuse) as scope:

                                        net = tf.expand_dims(inputs, -1) #input needs to be in the format NHWC!! if there is only one channel, expand it by 1 dimension

                                        net = slim.conv2d(net, 64, [3, 9], scope='conv1')
                                        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool1')
                                        net = slim.conv2d(net, 128, [3, 5], scope='conv2')
                                        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
                                        net = slim.flatten(net)
                                        net = slim.fully_connected(net, 256, scope='fc1')
                                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')
                                        logits = slim.fully_connected(net, num_classes, scope='fc2', activation_fn=None)

                                        return logits



def loss_fkt(logits, y):
        return tf.reduce_mean(tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = y, label_smoothing=0.05)) 



def build_model(x, 
		y,
	        num_classes=2,
		num_estimator=32,
                subsample=0.25,
                is_training=True,
		reuse=None
		):
        """
	 handle model. calculate the loss and the prediction for some input x and the corresponding labels y
	 input: x shape=[None,bands,frames,num_channels], y shape=[None]
	 output: loss shape=(1), prediction shape=[None]

        """
        #preprocess
        y = slim.one_hot_encoding(y, num_classes)
        
        loss = 0
        predictions=y * 0
        batch_size = x.get_shape()[0].value

        #models
        for i in range(num_estimator):
                #sample from minibatch - instead of bootstrap / TODO something better?
                idx = np.random.randint(batch_size, size=(int(round(batch_size * subsample)),))
                bx = tf.gather(x,idx)
                by = tf.gather(y,idx)

                logits = classify(bx, num_estimator=num_estimator, num_classes=num_classes, is_training=is_training, reuse=reuse, scope='H%d'%i)
                loss += loss_fkt(logits, by)

                #majority vote
                if not is_training:
                      logits = classify(x, num_estimator=num_estimator, num_classes=num_classes, is_training=is_training, reuse=True, scope='H%d'%i)
                      predictions+=slim.one_hot_encoding(tf.argmax(slim.softmax(logits),1), num_classes)

        predictions = tf.argmax(predictions, 1)             

        return loss, predictions 	




#Parameters
TRAINABLE_SCOPES = None 



