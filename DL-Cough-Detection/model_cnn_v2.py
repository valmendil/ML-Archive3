#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import simple_arg_scope, batchnorm_arg_scope


"""
	larger conv net - train all
"""


def classify(inputs, 
	     num_classes,
             dropout_keep_prob=0.5,
	     scope='cnn_v3',
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
                           with tf.variable_scope(scope, [inputs], reuse=reuse) as scope:

                                        net = tf.expand_dims(inputs, -1) #input needs to be in the format NHWC!! if there is only one channel, expand it by 1 dimension
                                        print ('model input shape: %s'%net.get_shape())

                                        net = slim.repeat(net, 2, slim.conv2d, 64, [3, 9], scope='conv1')
                                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                                        net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv2')
                                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                                        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                                        net = slim.max_pool2d(net, [2, 2], scope='pool3')
                                        net = slim.flatten(net)
                                        net = slim.fully_connected(net, 4096, scope='fc1')
                                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								 scope='dropout1')
                                        net = slim.fully_connected(net, 4096, scope='fc2')
                                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
								 scope='dropout2')
                                        logits = slim.fully_connected(net, num_classes, scope='fc3', activation_fn=None)

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



