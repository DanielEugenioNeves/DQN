from arch_model import Mymodel
from buffer_replay import Buffer
import tensorflow as tf
from tensorflow.keras import layers,Model,optimizers,losses
from arch_model import Mymodel
from random import uniform
import random
import numpy as np
from collections import deque
import os

class Agent_DQN:

    def __init__(self, num_actions, input_dims):
        # Info of enviroment
        self.num_actions = num_actions
        self.input_dims  = input_dims
        
        # RMSPROP
        self.learning_rate = .00025
        self.gradient_momentum = .95
        self.squared_gradient_momentum = .95
        self.min_squared_gradient = .01

        # Q-Learning
        self.discount_factor = .99
        self.init_exploration_rate = 1.0
        self.final_exploration_rate = .01
        self.exploration_rate = self.init_exploration_rate
        self.final_exploration_frame = 9000000
        self.rate = (self.init_exploration_rate/self.final_exploration_frame)*20

        # Replay memory
        self.minibatch_size = 32
        self.replay_memory_size = 1000000
        self.start_train = 50000
        self.memory_replay = Buffer(self.replay_memory_size,self.minibatch_size)

        # Neural Network
        self.q        = Mymodel(self.input_dims,self.num_actions)._build_model()
        self.q_target = Mymodel(self.input_dims,self.num_actions)._build_model()
        self.target_update_frequency = 10000
        self.update_frequency        = 4
        self.history_lengh           = 4
        self.optimizer  = optimizers.RMSprop(learning_rate=.00025,rho=0.95,momentum=0.95,epsilon=0.01)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    
    def compute_loss(self):
        one_hot_actions = tf.keras.utils.to_categorical(self.actions, self.num_actions, dtype=np.float32)
        q = tf.reduce_sum(tf.multiply(self.q(self.states,training=True),one_hot_actions),axis=1)
        self.loss = self.huber_loss(self.target, q)
        return self.loss

    def compute_dqn_target(self):
                
        # Predict
        q_target_vals = self.q_target(self.next_states,training=False)
        
        # Target calculation
        y =  (self.rewards + (1-self.terminateds) * self.discount_factor*np.amax(q_target_vals,axis=1))
        y_pred = tf.Variable(y,dtype=tf.float32, shape=(32,))
        return y_pred

    def compute_ddqn_target(self):

        # Predict
        argmax_q = np.argmax(self.q(self.next_states,training=False) ,axis=1)
        qt_vals = np.take(self.q_target(self.next_states,training=False),argmax_q,axis=None)        
        # Target calculation
        y =  (self.rewards + (1-self.terminateds) * self.discount_factor*qt_vals)
        y_pred = tf.Variable(y,dtype=tf.float32, shape=(32,))
        return y_pred

    def train_agent_raw(self,minibatch,method):
        # Experience(state,action,reward,next_state,done)
        """
        minibatch_array configuration
            0 : state
            1 : action
            2 : reward
            3 : next_state
            4 : terminated
        """ 
        minibatch_array = np.asarray(minibatch)

        # Get states
        self.states = np.stack(minibatch_array[...,0])
        # Get action
        self.actions = np.stack(minibatch_array[...,1])
        #Get rewards
        self.rewards = minibatch_array[...,2]
        # Get next_states
        self.next_states = np.stack(minibatch_array[...,3])      
        #Get information about the game over
        self.terminateds = minibatch_array[...,4]
        #Loss
        self.loss = .0

        if method == 'ddqn':
            self.target = tf.stop_gradient(self.compute_ddqn_target())
        elif method == 'dqn':
            self.target = tf.stop_gradient(self.compute_dqn_target())

        one_hot_actions = tf.keras.utils.to_categorical(self.actions, self.num_actions, dtype=np.float32)
        
        with tf.GradientTape() as tape:
            q_vals = self.q(self.states,training=True)
            q = tf.reduce_sum(tf.multiply(q_vals,one_hot_actions),axis=1)
            self.loss = tf.reduce_mean(self.huber_loss(self.target, q))
        
        # gradients = tape.gradient(self.loss, self.q.trainable_variables)
        # gradients = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in gradients]
        # self.optimizer.apply_gradients(zip(gradients, self.q.trainable_variables))
        gradients       = tape.gradient(self.loss, self.q.trainable_variables)
        processed_grads = [g for g in gradients]
        grads_and_vars  =  zip(gradients, self.q.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)
    
        return self.loss

        
    def train_agent(self,minibatch,method):
        # Experience(state,action,reward,next_state,done)
        """
        minibatch_array configuration
            0 : state
            1 : action
            2 : reward
            3 : next_state
            4 : terminated
        """ 
        minibatch_array = np.asarray(minibatch)

        # Get states
        self.states = np.stack(np.divide(minibatch_array[...,0],255))
        # Get action
        self.actions = np.stack(minibatch_array[...,1])
        #Get rewards
        self.rewards = minibatch_array[...,2]
        # Get next_states
        self.next_states = np.stack(np.divide(minibatch_array[...,3],255)) 
        #Get information about the game over
        self.terminateds = minibatch_array[...,4]
        #Loss
        #self.loss = .0

        if method == 'ddqn':
            self.target = tf.stop_gradient(self.compute_ddqn_target())
        elif method == 'dqn':
            self.target = tf.stop_gradient(self.compute_dqn_target())

        t_loss      = lambda : tf.reduce_mean(self.compute_loss())
        var_list    = lambda : self.q.trainable_weights
        self.optimizer.minimize(t_loss, var_list)
        return tf.reduce_mean(self.loss)

    def choose_action(self,mode,state):
        type_action = 'q_net'
        rng = uniform(0,1)
        q_vals = self.q(state)
        if mode == 'train':
            if rng < self.exploration_rate:
                type_action = 'rand'
                return random.randint(0,(self.num_actions-1)),type_action
            return np.argmax(q_vals),type_action
        elif mode ==  'evaluation':
            if rng < 0.05:
                type_action = 'rand'
                return random.randint(0,(self.num_actions-1)),type_action
            return np.argmax(q_vals),type_action
    
    def update_weights_q_target(self):
        for layer_q_target,layer_q in zip(self.q_target.layers,self.q.layers):
            w_q        = layer_q.get_weights()
            w_q_target = layer_q_target.get_weights()
            layer_q_target.set_weights(w_q)
            
    def update_epsilon(self):
        if self.exploration_rate > self.final_exploration_rate:
            self.exploration_rate = self.exploration_rate - self.rate
        else:
            self.exploration_rate = self.final_exploration_rate

    def store_experience(self,state,action,reward,next_state,terminated):
        self.memory_replay.store(state,action,reward,next_state,terminated)
        self.update_epsilon()
    
    def save_model_weights(self,folder_train,frame_number):
        calculated = folder_train + '/checkpoint' + str(frame_number)
        self.q.save_weights(calculated)
        
    def load_model_weights(self,folder_train,frame_number):
        calculated = folder_train + '/checkpoint' + str(frame_number)
        self.q.load_weights(calculated)