"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

Using:
Tensorflow: 2.0
gym: 0.7.3
# Modified by Qizhi He (qizhi.he@pnnl.gov)
# Ref to https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification
"""

import numpy as np
# import pandas as pd
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)
# tf.random.set_seed(1) # for tf2

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.8,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment_ref=0.01,
            e_greedy_increment=0.01,
            e_greedy_increment_type = 'fixed',
            output_graph=False,
            randomness=3,
            e_greedy_increment_ratios = 1,
            e_greedy_increment_intervals = [0,1]
            ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment_ref = e_greedy_increment_ref
        self.epsilon_increment = self.e_greedy_increment_ref
        self.e_greedy_increment_type = e_greedy_increment_type
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.randomness = randomness
        self.e_greedy_increment_ratios = e_greedy_increment_ratios
        self.e_greedy_increment_intervals = e_greedy_increment_intervals

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        #self.memory = np.zeros((self.memory_size, n_features * 2 + 2 ))
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2 + 1))    # Yunxiang: +1?

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 100, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
                
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
            #print("n_l1=========",n_l1)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r,ddone, s_): #Jie...      add "done"
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r,ddone], s_)) #Jie...      add "done"
        #transition = np.hstack((s, [a, r], s_))
        #print("transition")
        #print(transition)
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size

        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if self.epsilon>1.999:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            k=self.randomness
            #print("actaions_value ",actions_value)
            max5=np.argpartition(actions_value[0,:], len(actions_value[0,:]) - k)[-k:]
            #print("max5",max5)
            pick_one=np.random.randint(0,k)
            action=max5[pick_one]
        elif np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            #print(actions_value)
            #print("action ",action,actions_value.shape)
        else:
            # self.n_actions
            # actions_value = np.array([[0,0,0]])
            actions_value = np.zeros((1, self.n_actions))
            action = np.random.randint(0, self.n_actions)
        return action, actions_value

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        ddone = batch_memory[:, self.n_features + 2] #Jie...      load information "done"
        # if not ddone: #Jie...      check done
        #     q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # else: #Jie...
        #     q_target[batch_index, eval_act_index] = reward #Jie...
        # Yunxiang modified.
        idt = np.where(ddone)
        idf = np.where(np.logical_not(ddone))
        q_target[idt,eval_act_index[idt]] = reward[idt]
        q_target[idf,eval_act_index[idf]] = reward[idf] + self.gamma*np.max(q_next[idf[0],:],axis=1)
        
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # Update epsilon increment step based on epsilon value.
        e_greedy_increment_ratios = self.e_greedy_increment_ratios
        e_greedy_increment_intervals =self.e_greedy_increment_intervals
        self.e_greedy_increment = self.e_greedy_increment_ref
        if self.e_greedy_increment_type.lower() == 'variable':
            for i in range(len(e_greedy_increment_ratios)):
                if self.epsilon>e_greedy_increment_intervals[i] and self.epsilon<=e_greedy_increment_intervals[i+1]:
                   self.e_greedy_increment =  self.e_greedy_increment_ref/e_greedy_increment_ratios[i]
                   break

        # increasing epsilon
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self,dir_file,model_index=1):
        import matplotlib.pyplot as plt
        import os
        plt.figure(8)
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig(dir_file+'Cost_'+str(model_index)+'.png')
        plt.show if os.name == 'nt' else plt.close(8)

        tmp_array = np.array(self.cost_his)
        np.savetxt(dir_file+'training_cost_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
        