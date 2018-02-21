#Author: Kevin Kipfer

import librosa
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from random import shuffle
import threading

try:
       import librosa
except:
       print("pip install librosa ; if you want mfcc_batch_generator")



from preprocessing import *
from utils import find_files



def data_iterator(data, 
                  load_batch_size,
       	  is_training): 		
       """ iterate through the data
       	input: data (list with paths), 
       	       load_batch size (integer - how many should be loaded at once)
       	output: data_batch, label_batch """
       
       batch_idx = 0
       assert load_batch_size % 2 == 0

       '''
       	idxs = np.arange(0, len(data))
       	np.random.shuffle(idxs)
       	shuf_paths = data[idxs]
       	shuf_labels = labels[idxs]
       '''
       
       #sample balanced batches
       load_batch_size_half = load_batch_size // 2
       #create label dummy: 1st half true, 2nd half false
       labels_batch = np.array([1 if x < load_batch_size_half else 0 for x in range(load_batch_size)]) 
       #max samples per epoch: assumption data[0] (true) and data[1] (false) are similarly large
       batches_per_epoch = min(len(data[0]),len(data[1])) - load_batch_size_half - 1 
       while True:
	       	shuf_cough = data[0]
	       	shuf_other = data[1]
	       	np.random.shuffle(shuf_cough) 
	       	np.random.shuffle(shuf_other)
	       	for batch_idx in range(0, batches_per_epoch, load_batch_size_half):
	       	    end = batch_idx + load_batch_size_half
	       	    cough_batch = list(shuf_cough[batch_idx:end])
	       	    other_batch = list(shuf_other[batch_idx:end])
	       	    cough_batch.extend(other_batch)
	       	    data_batch = fetch_samples(cough_batch, is_training)
	       	    data_batch = data_batch.astype("float32")
	       	    #print ('next batch')
	       	    if data_batch.shape[0] != load_batch_size:
                          raise ValueError('the data_iterator produced a batch of size [%d]; expecteds size [%d]; ( out batch shape: %s; length of inputs to fetch_samples: %d)' \
                                               %(data_batch.shape[0],load_batch_size, str(data_batch.shape), len(cough_batch)))
	       	    yield data_batch, labels_batch



class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    required input: data -> a tuple (list of paths, list of labels)
    optional input: is_training -> should data augmentation be used?
       	    etc.		    
    """
    def __init__(self,
       	data,
       	batch_size = 32,
       	load_batch_size = 32,
       	is_training = True,
       	capacity=1000,
       	min_after_dequeue=500,
        show_input_shape=True):

       self.batch_size = batch_size
       self.load_batch_size = load_batch_size
       self.data = data
       self.is_training = is_training

       #find input shape size
       shape = list(fetch_samples([data[0][0]])[0].shape)		
       if show_input_shape:
               print ( 'the input to the model placeholder has shape %s '% shape)					
       

       # The actual queue of data. The queue contains a vector for
       # the features, and a scalar label.
       #self.queue = tf.RandomShuffleQueue(shapes=[[bands,frames,num_channels], []],        		
       self.queue = tf.RandomShuffleQueue(shapes=[shape, []], 					
                                           dtypes=[tf.float32, tf.int64],
                                           capacity=capacity,
                                           min_after_dequeue=min_after_dequeue)

       #placeholders for reading the data
       #self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, bands,frames,num_channels])      
       self.dataX = tf.placeholder(dtype=tf.float32, shape=([None] + shape) )  			
       self.dataY = tf.placeholder(dtype=tf.int64, shape=[None, ])

       # The symbolic operation to add data to the queue
       self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])

       self.stop_event= threading.Event()

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(self.batch_size)
        return images_batch, labels_batch

    def thread_main(self, sess, stop_event):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY in data_iterator(self.data, load_batch_size = self.load_batch_size, is_training = self.is_training):
            if stop_event.is_set(): 
               break
            sess.run(self.enqueue_op, feed_dict={self.dataX:dataX, self.dataY:dataY})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,self.stop_event))
            t.daemon = True # thread will close when parent quits 
            t.start()
            threads.append(t)
        return threads

    def close(self):
        """ Close background threads without closing the session """
        self.stop_event.set()


       
