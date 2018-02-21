#Author: Kevin Kipfer

import librosa
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, fnmatch, sys
#plt.style.use('ggplot')





def simple_arg_scope(weight_decay=0.0005, 
       	             seed=0,
                     activation_fn=tf.nn.relu ):
  """Defines a simple arg scope.
       relu, xavier, 0 bias, conv2d padding Same, weight decay
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
             	      #weights_initializer= slim.variance_scaling_initializer(seed=seed),
             	      weights_initializer= tf.contrib.layers.xavier_initializer(seed=seed),# this is actually not needed
             	      activation_fn=activation_fn,
                      weights_regularizer= slim.l2_regularizer(weight_decay) if weight_decay is not None else None,
                      biases_initializer=tf.zeros_initializer()):
           with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
             	return arg_sc


def batchnorm_arg_scope(batch_norm_decay=0.997,
                       batch_norm_epsilon=1e-5,
                       batch_norm_scale=True):
  """Defines an arg_scope that initializes all the necessary parameters for the batch_norm
     add this if you want to use the batch norm 
  Returns:
    An arg_scope.
  """

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      normalizer_fn=slim.batch_norm, 
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
        return arg_sc


def clip_grads(grads_and_vars, clipper=5.):
    with tf.name_scope('clip_gradients'):
         gvs = [(tf.clip_by_norm(grad, clipper), val) for grad,val in grads_and_vars]
         return gvs

def add_grad_noise(grads_and_vars, grad_noise=0.):
    with tf.name_scope('add_gradients_noise'):
         gvs = [(tf.add(grad, tf.random_normal(tf.shape(grad),stddev=grad_noise)), val) for grad,val in grads_and_vars]
         return gvs

class HiddenPrints:
    """
       hide console outputs
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

 
def get_variables_to_train(trainable_scopes=None, show_variables=False):
    """Returns a list of variables to train.

      Returns:
        A list of variables to train by the optimizer.
    """
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
    if trainable_scopes is None:
        return trainable_variables

    variables_to_train = []

    if show_variables:
           print ('*********************************************************************')
           print ('trainable variables: ')
    for s in trainable_scopes:
           for v in trainable_variables:
              if s in v.name:
                        variables_to_train.append(v)
                        if show_variables:
                               print (v.name)

    print ('*********************************************************************')

    return variables_to_train


def load_model(sess, 
       	checkpoint_path, 
       	show_cp_content=True, 
       	ignore_missing_vars=False):
        """warm-start the training.
        """
       
        if not os.path.exists(checkpoint_path):
       		os.makedirs(checkpoint_path)  

        latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)	
        if not latest_ckpt:
               return tf.train.Saver()
	
        print ( 'restore from checkpoint: '+checkpoint_path )

        with HiddenPrints():
                variables = slim.get_model_variables() # slim.get_variables_to_restore()
        
        if show_cp_content:
                print ()
                print ('------------------------------------------------------------------------------')
                print ('variables stored in checkpoint:')
                from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
                print_tensors_in_checkpoint_file(latest_ckpt, '', False)
                print ('------------------------------------------------------------------------------')
       	
        if ignore_missing_vars:
       		reader = tf.train.NewCheckpointReader(latest_ckpt)
	       	saved_shapes = reader.get_variable_to_shape_map()

	       	var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables
	       	                            if var.name.split(':')[0] in saved_shapes])

	       	print ('nr available vars in the checkpoint: %d'%len(var_names))
	       	restore_vars = []
	       	name2var = dict(zip(map(lambda x:x.name.split(':')[0], variables), variables))
	       	with tf.variable_scope('', reuse=True):
	       	            for var_name, saved_var_name in var_names:
	       	                curr_var = name2var[saved_var_name]
	       	                var_shape = curr_var.get_shape().as_list()
	       	                if var_shape == saved_shapes[saved_var_name]:
	       	                    restore_vars.append(curr_var)

	       	print ('nr vars restored: %d'%len(restore_vars))    
	       	saver = tf.train.Saver(restore_vars)
        else:
                saver = tf.train.Saver(variables)

        saver.restore(sess,latest_ckpt)
        return saver



def find_files(root, fntype, recursively=False):
       fntype = '*.'+fntype
       
       if not recursively:
       		return glob.glob(os.path.join(root, fntype))
       
       matches = []
       for dirname, subdirnames, filenames in os.walk(root):
           for filename in fnmatch.filter(filenames, fntype):
                matches.append(os.path.join(dirname, filename))
       return matches



       
