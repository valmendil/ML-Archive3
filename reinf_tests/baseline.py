import gym
import tensorflow as tf
import numpy as np
import cv2
import sys, os, shutil
import random
import time
from collections import deque
from collections import defaultdict
from tensorflow.models.image.mnist.convolutional import NUM_EPOCHS

# https://gym.openai.com/envs#atari

# stop program by pressing the key escape !!

GAME = 'Pong-v0'
RESULT_DIR = 'tmp/records'
MODEL_DIR = 'tmp/saved_networks/'
SUMMARY_DIR = 'tmp/summary'


RENDER_SCREEN = False
SAVE_ALL_VIDEOS = False #only usable if RENDER_SCREEN is true


DO_TRAINING = True

NR_EPISODES = 1500000
NR_ITERATIONS = 1000 # just play

###########################################################

GAMMA = 0.99  # decay rate of past observations
OBSERVE = 500.  # timesteps to observe before training
EXPLORE = 1000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.05  # final value of epsilon
INITIAL_EPSILON = 1.0  # starting value of epsilon
REPLAY_MEMORY = 100000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
K = 1  # only select an action every Kth frame, repeat prev for others

sizeX = 80
sizeY = 80

###########################################################



human_wants_stop = False


def key_press(key, mod):
    global human_wants_stop
    if key == 65307:
        human_wants_stop = True


def restore(sess):
    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    return saver


def createNetwork(nr_actions, summary=False):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, nr_actions])
    b_fc2 = bias_variable([nr_actions])

    # input layer
    s = tf.placeholder("float", [None, sizeX, sizeY, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    
    return s, readout, h_fc1


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def trainNetwork(sess, env, observation_space, action_space, nr_actions):
    summary_writer = tf.train.SummaryWriter(SUMMARY_DIR)
    
    # Summary
    score = tf.placeholder(tf.float32, None, name="score")
    score_summary =  tf.scalar_summary("Score (Sum of the Rewards per Episode)", score)
    avgreward = tf.placeholder(tf.float32, None, name="avgreward")
    avgreward_summary =  tf.scalar_summary("Average Reward per Episode", avgreward)
    avgq = tf.placeholder(tf.float32, None, name="avgq")
    avgq_summary =  tf.scalar_summary("Average Action Value (Q)", avgq)
    numep = tf.placeholder(tf.int32, None, name="numep")
    num_ep_summary = tf.scalar_summary("Number Of Actions Per Episode", numep)
    
    summary_op = tf.merge_all_summaries()
    
    
    s, readout, h_fc1 = createNetwork(nr_actions, summary=True)
    
    epsilon = INITIAL_EPSILON
    t = 0
    D = deque()

    # define the cost function
    a = tf.placeholder("float", [None, nr_actions])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    
    cost_summary =  tf.scalar_summary("Cost per MiniBatch", cost)

    # saving and loading networks
    saver = restore(sess)


    step=0 
    
    for i_episode in range(NR_EPISODES): 

        if RENDER_SCREEN:
            if SAVE_ALL_VIDEOS:
                env.monitor.start(RESULT_DIR + "/" + str(i_episode))
            else:
                env.monitor.start(RESULT_DIR, force=True)

        # get the first state and preprocess the image to sizeXxsizeYx4
        obs = env.reset()

        x_t = cv2.cvtColor(cv2.resize(obs, (sizeX, sizeY)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        
        #summary variables
        num_eps = 0
        total_rewards = 0
        total_q = 0

        while True:
            if RENDER_SCREEN:
                env.render()
                env.viewer.window.on_key_press = key_press
                if human_wants_stop:
                    env.monitor.close()
                    saver.save(sess, MODEL_DIR + GAME + '-dqn', global_step=t)
                    return
                
            num_eps += 1
            step+=1

            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s: [s_t]})[0]
            a_t = np.zeros([nr_actions])
            action_index = 0
            if random.random() <= epsilon or t <= OBSERVE:
                action_index = action_space.sample()
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1

            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observe next state and reward
            obsNew, r_t, done, info = env.step(action_index, is_training=True)
            x_t1 = cv2.cvtColor(cv2.resize(obsNew, (sizeX, sizeY)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)

            x_t1 = np.reshape(x_t1, (sizeX, sizeY, 1))
            s_t1 = np.append(x_t1, s_t[:, :, 0:3], axis=2)

            total_rewards += r_t

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, done))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            if done:
                break

            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                # get the batch variables (screen, action, reward)
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
                for i in range(0, len(minibatch)):
                    # if done only equals reward
                    if minibatch[i][4]:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                if step % 100:
                    c_sum = cost_summary.eval(feed_dict={
                        y: [y_batch[0]],
                        a: [a_batch[0]],
                        s: [s_j_batch[0]]})
                    summary_writer.add_summary(c_sum, step)

                # perform gradient step
                train_step.run(feed_dict={
                    y: y_batch,
                    a: a_batch,
                    s: s_j_batch})


                total_q += y_batch[0]
                
            # update the old values
            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, MODEL_DIR + GAME + '-dqn', global_step=t)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
            print "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(
                readout_t)

        if RENDER_SCREEN:
            env.monitor.close()
        
        summary = sess.run(summary_op, feed_dict={score: total_rewards, numep: num_eps, avgreward: (total_rewards/num_eps), avgq: (total_q/num_eps)})
        summary_writer.add_summary(summary, i_episode)
        
    saver.save(sess, MODEL_DIR + GAME + '-dqn', global_step=t)
        


def play(sess, env, observation_space, action_space, nr_actions):
    t = 0

    s, readout, h_fc1 = createNetwork(nr_actions)
    restore(sess)

    env.monitor.start(RESULT_DIR, force=True)

    # get the first state and preprocess the image to sizeXxsizeYx4
    obs = env.reset()

    x_t = cv2.cvtColor(cv2.resize(obs, (sizeX, sizeY)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    for nr_iterations in range(NR_ITERATIONS):
        env.render()
        env.viewer.window.on_key_press = key_press

        if human_wants_stop:
            env.monitor.close()
            saver.save(sess, MODEL_DIR + GAME + '-dqn', global_step=t)
            return

        # choose an action
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([nr_actions])
        action_index = np.argmax(readout_t)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        obsNew, r_t, done, info = env.step(action_index, is_training=False)
        x_t1 = cv2.cvtColor(cv2.resize(obsNew, (sizeX, sizeY)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)

        x_t1 = np.reshape(x_t1, (sizeX, sizeY, 1))
        s_t1 = np.append(x_t1, s_t[:, :, 0:3], axis=2)

        if done:
            break

        # update the old values
        s_t = s_t1
        t += 1

        # print info
        print "TIMESTEP", t, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

    env.monitor.close()


if __name__ == '__main__':

    shutil.rmtree(RESULT_DIR, ignore_errors=True)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

        # initialize game state
    env = gym.make(GAME)

    observation_space = env.observation_space
    action_space = env.action_space
    nr_actions = action_space.n

    sess = tf.InteractiveSession()

    if DO_TRAINING:
        trainNetwork(sess, env, observation_space, action_space, nr_actions)
    else:
        play(sess, env, observation_space, action_space, nr_actions)
