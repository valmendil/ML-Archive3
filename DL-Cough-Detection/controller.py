#!/usr/bin/python
#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import *
from input_pipeline import *


#******************************************************************************************************************

#from model_cnn_weak import *
#from model_cnn_v2 import *
#from model_cnn_v3_3 import *
#from model_cnn_v4 import *
#from model_resnet_v1 import *
#from model_densenet_v1 import *
from model_boost_v4_5 import *
#from model_boost_v4_6 import *
#from model_bag_v1 import *
#from model_boost_v1 import *
#from model_rnn_v2 import *
#from model_rnn_v1 import *

#******************************************************************************************************************


ROOT_DIR = './Audio_Data'


def train(train_data,
         test_data,
         num_classes=2,
         eta=2e-3, #learning rate
         grad_noise=1e-3,
         clipper=10.,
         #checkpoint_dir='./checkpoints/test',
         #checkpoint_dir='./checkpoints/cnn_v1.02',
         #checkpoint_dir='./checkpoints/cnn_v2.9',
         #checkpoint_dir='./checkpoints/cnn_v3.3x1',
         #checkpoint_dir='./checkpoints/rnn_v1.03',
         #checkpoint_dir='./checkpoints/rnn_v2.01',
         #checkpoint_dir='./checkpoints/resnet_v1.0',
         #checkpoint_dir='./checkpoints/dense_v1.0',
         checkpoint_dir='./checkpoints/boost_v4.5x',
         #checkpoint_dir='./checkpoints/weak_v1.1k',
         #checkpoint_dir='./checkpoints/boost_v1.0',
         batch_size=64,
         n_producer_threads=12,
         trainable_scopes=TRAINABLE_SCOPES,
         train_capacity=7500,
         test_capacity=1500,
         max_steps = 500000,
         log_every_n_steps=200,
         eval_every_n_steps=100,
         save_every_n_steps=2000,
         save_checkpoint=True):


       print ('save checkpoints to: %s'%checkpoint_dir)

       graph = tf.Graph() 
       with graph.as_default():
              #load training data
              with tf.device("/cpu:0"):
                    train_runner = CustomRunner(train_data, batch_size=batch_size, capacity=train_capacity)
                    train_batch, train_labels = train_runner.get_inputs()

              #initialize
              global_step = tf.Variable(0, name='global_step', trainable=False)
              eta = tf.train.exponential_decay(eta, global_step, 80000, 0.96, staircase=False) 
              train_op = tf.train.AdamOptimizer(learning_rate=eta) 

              train_loss, preds = build_model(train_batch, train_labels)
              tf.summary.scalar('training/train_loss', train_loss )
	
              #add regularization
              regularization_loss = tf.losses.get_regularization_losses() #use tf.losses.get_regularization_loss instead?
              if regularization_loss: 
                        train_loss += tf.add_n(regularization_loss)
                        tf.summary.scalar('training/total_loss', train_loss )
             
              #specify what parameters should be trained
              params = get_variables_to_train(trainable_scopes) 
              print ('nr trainable vars: %d'%len(params))  

              #control depenencies for batchnorm, ema, etc. + update global step
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
              with tf.control_dependencies(update_ops):
                        # Calculate the gradients for the batch of data.
                        grads = train_op.compute_gradients(train_loss, var_list = params)   
                        # gradient clipping
                        grads = clip_grads(grads, clipper=clipper)
                        # add noise
                        if grad_noise > 0:
                                grad_noise = tf.train.exponential_decay(grad_noise, global_step, 10000, 0.96, staircase=False) 
                                grads = add_grad_noise(grads, grad_noise)
                        # minimize
                        train_op = train_op.apply_gradients(grads, global_step=global_step)
                      
              #some summaries
              tf.summary.scalar('other/learning_rate', eta  )
              tf.summary.scalar('other/gradient_noise', grad_noise  )
              
              with tf.variable_scope('gradients'):
              	for grad, var in grads:
              		if grad is not None:
              			tf.summary.histogram(var.op.name, grad)    
  		       
              #collect summaries
              summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

              #Merge all train summaries.
              summary_op = tf.summary.merge(list(summaries), name='summary_op')

              #load Test Data
              with tf.device("/cpu:0"):
                   test_runner = CustomRunner(test_data, is_training = False, batch_size=batch_size, capacity=test_capacity)
                   test_batch, test_labels = test_runner.get_inputs()

              #Evaluation
              test_loss, predictions = build_model(test_batch, test_labels, is_training=False, reuse=True)	

              #Collect test summaries
              with tf.name_scope('evaluation' ) as eval_scope:
                      tf.summary.scalar('test_loss', test_loss )

                      mpc, mpc_update = tf.metrics.mean_per_class_accuracy(predictions=predictions, labels=test_labels, num_classes=num_classes)
                      tf.summary.scalar('mpc_accuracy', mpc )

                      accuracy, acc_update = tf.metrics.accuracy(predictions=predictions, labels=test_labels)
                      tf.summary.scalar('accuracy', accuracy )

                      auc, auc_update = tf.metrics.auc(labels=test_labels, predictions=predictions)
                      tf.summary.scalar('AUC', auc )
                      
                      precision, prec_update = tf.metrics.precision(labels=test_labels, predictions=predictions)
                      tf.summary.scalar('precision', precision )
                      
                      recall, rec_update = tf.metrics.recall(labels=test_labels, predictions=predictions)
                      tf.summary.scalar('recall', recall )
                      

                      #tf.summary.image('test_batch', tf.expand_dims(test_batch, -1))
                      #tf.summary.histogram('predictions', predictions)
                      #tf.summary.histogram('labels', test_labels)

              test_summary_op = tf.summary.merge(list(tf.get_collection(tf.GraphKeys.SUMMARIES, eval_scope)), name='test_summary_op')
              test_summary_update = tf.group(acc_update, mpc_update, auc_update, prec_update, rec_update)

              #initialize
              gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
              sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options))
              init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

              with sess.as_default():
                sess.run(init)

              	#checkpoints              
                saver = load_model(sess, checkpoint_dir)

                # start the tensorflow QueueRunner's
                tf.train.start_queue_runners(sess=sess)

                # start our custom queue runner's threads
                train_runner.start_threads(sess, n_threads=n_producer_threads)
                test_runner.start_threads(sess, n_threads=1)

                #wait for the queues to be filled
                time.sleep(20) 
              		    
                train_writer = tf.summary.FileWriter(checkpoint_dir+"/train", sess.graph)
                test_writer = tf.summary.FileWriter(checkpoint_dir+"/test")

                #assert that no new tensors get added to the graph after this steps 
                sess.graph.finalize()

                print ('start learning')
                try:
              	        for i in range(max_steps): 
              		        #training
                                _, step, train_loss_ = sess.run([train_op, global_step, train_loss])
              			#logging: update training summary
                                if i >= 300 and i%(log_every_n_steps) == 0:
                                        summary = sess.run([summary_op])[0]
                                        train_writer.add_summary(summary, step)
                           
              			#logging: update testing summary
                                if i >= 300 and i%(eval_every_n_steps) == 0:
                                        summary, mpc_, accuracy_, _ = sess.run([test_summary_op, mpc, accuracy, test_summary_update])
                                        print ('EVAL: step: %d, idx: %d, mpc: %f, accuracy: %f'% (step, i,  mpc_, accuracy_))
                                        test_writer.add_summary(summary, step)
                           
                                #save checkpoint
                                if i%(save_every_n_steps) == save_every_n_steps-1 and save_checkpoint:
                                        print ('save model (step %d)'%step)
                                        saver.save(sess,checkpoint_dir+'/checkpoints', global_step=step)

                except KeyboardInterrupt:
                      	        print("Manual interrupt occurred.")

                train_runner.close()
                mpc_, accuracy_, loss_ = sess.run([mpc, accuracy, test_loss])

                print ('################################################################################')
                print ('Results - mpca:%f, accuracy:%f, loss:%f'%(mpc_,accuracy_,loss_))
                print ('################################################################################')
        
                test_runner.close()
                sess.close()


    
def main(unused_args):

       listOfParticipantsToExcludeInTrainset = ["p05", "p17", "p34", "p20", "p28", "p09", "p08", "p11", "p31", "p21", "p14"] #participants used in the test-set

       list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
                               '04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']

       ##
       # READING COUGH DATA
       #
       #

       print ('use data from root path %s'%ROOT_DIR)

       coughAll = find_files(ROOT_DIR + "/04_Coughing", "wav", recursively=True)
       assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'

       #remove broken files
       for broken_file in list_of_broken_files:
           broken_file = os.path.join(ROOT_DIR, broken_file)
           if broken_file in coughAll:
                 print ( 'file ignored: %s'%broken_file )
                 coughAll.remove(broken_file)

       #split cough files into test- and training-set
       testListCough = []
       trainListCough = coughAll
       for name in coughAll:
           for nameToExclude in listOfParticipantsToExcludeInTrainset:
              if nameToExclude in name:
                  testListCough.append(name)
                  trainListCough.remove(name)

       print('nr of samples coughing: %d' % len(testListCough))

       ##
       # READING OTHER DATA
       #
       #

       other = find_files(ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)

       testListOther = []
       trainListOther = other
       for name in other:
           for nameToExclude in listOfParticipantsToExcludeInTrainset:
              if nameToExclude in name:
                  testListOther.append(name)
                  trainListOther.remove(name)

       print('nr of samples NOT coughing: %d' % len(testListOther))


       train_data = (trainListCough, trainListOther)
       test_data = (testListCough, testListOther)


       ##
       # START TRAINING
       #
       #

       tf.set_random_seed(0)
       train(train_data, test_data)
    



if __name__ == '__main__':
       tf.app.run()    


