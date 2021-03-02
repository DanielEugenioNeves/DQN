import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from ale.ale_python_interface import ALEInterface
from preprocessor import State_Preprocessor
from arch_model import Mymodel
from DQN import Agent_DQN

import csv
import datetime
from collections import deque

import os
import signal

from random import randrange


class Enviroment:

    def __init__(self):

        # Image configs
        self.WIDTH = 84
        self.HEIGHT = 84
        self.FRAMES = 4
        self.input_dims = (self.WIDTH, self.HEIGHT, self.FRAMES)

        # Arcade Learning Enviromente (ALE)
        self.ale = ALEInterface()
        self.rom_name = ''
        self.rom_path = ''
        self.legal_actions = []
        self.num_actions = 0

        #
        self.MAX_FRAMES = 10000001
        self.loss = .0

        #
        self.folder_train = ''
        self.log_data = []
        
        self.head_file = True
        self.head_resume = True
    def set_hardware_to_train(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def set_enviroment_configurations(self):
        # ALE enviroment parameters
        self.ale.setInt(b'random_seed', 123)
        self.ale.setInt(b'frame_skip', 5)
        self.ale.setFloat(b'repeat_action_probability', .25)
        self.ale.setBool(b'color_averaging', True)
        self.ale.setBool(b'display_screen', self.display_screen)
        # ale.setBool(b'sound', True)

        # ROM Configs

        self.rom_path = 'rom/' + self.rom_name + '.bin'
        self.ale.loadROM(str.encode(self.rom_path))

        # Game mode
        self.ale.setMode(0)
        self.ale.setDifficulty(0)

        # Get the list of legal actions
        self.legal_actions = self.ale.getLegalActionSet()
        self.num_actions = len(self.ale.getLegalActionSet())

    

    def create_folder(self):
        # Experiments
        folder_exp = self.experiment
        if not os.path.isdir(folder_exp):
            os.mkdir(folder_exp)

        # Rom
        folder_game = folder_exp + self.rom_name + '/'
        if not os.path.isdir(folder_game):
            os.mkdir(folder_game)

        # Train
        name_test = 'train_agent_' + self.method
        self.folder_train = folder_game + name_test
        if not os.path.isdir(self.folder_train):
            os.mkdir(self.folder_train)

    def read_data(self):
        date = datetime.datetime.now().strftime('%x %X')
        data = self.train_step,date,self.rom_name,self.method,\
            self.flag_w,self.ale.getFrameNumber(),self.episode,\
            self.ale.getEpisodeFrameNumber(),self.immediate_reward,\
            self.agent.memory_replay.lenght(),self.loss,self.agent.exploration_rate
        self.log_data.append(data)

    def write_log_train(self):
        
        # File
        fieldnames = ['Train','date-time', 'rom', 'method', 'checkpoint', 'total_frame',
                      'episodes', 'frames_episode','immediate_reward','buffer_len','loss','epsilon']
        file_path = self.folder_train + '/new_data_train_step_' + str(self.train_step) + '.csv'
        
        with open(file_path, 'a') as arquivo_csv:
            f_in = csv.writer(arquivo_csv, delimiter=',', lineterminator='\n')
            if self.head_file:
                f_in.writerow(fieldnames)
                self.head_file = False
            for log in self.log_data:
                f_in.writerow(log)
            self.log_data.clear()

    def resume_episode(self):

        if self.loss != 0:
            avg_loss = self.total_loss/self.count_step_loss
        else:
            avg_loss = 0.0
        print('T_frames: {} | Ep: {} | Ep_frame: {} | T_reward: {} | AVG_loss: {:.10f} | Epsilon: {:.8f} | buffer: {}'.
              format(self.ale.getFrameNumber(), self.episode, self.ale.getEpisodeFrameNumber(), self.episode_reward, avg_loss, self.agent.exploration_rate, self.agent.memory_replay.lenght()))
        fieldnames = ['Train','date-time', 'rom', 'method', 'checkpoint','Episode', 'Total frames', 'Episode_frames', 'buff','Reward', 'td_erro','episilon']
        file_path = self.folder_train + '/data_resume_step_train_' + str(self.train_step) + '.csv'
        date = datetime.datetime.now().strftime('%x %X')
        data = (self.train_step,date, self.rom_name, self.method, self.checkpoint, self.episode,
                self.ale.getFrameNumber(),self.ale.getEpisodeFrameNumber(), self.agent.memory_replay.lenght(),self.episode_reward, avg_loss, self.agent.exploration_rate)
        with open(file_path, 'a', encoding='utf-8') as arquivo_csv:
            f_in = csv.writer(arquivo_csv, delimiter=',',
                              lineterminator='\n')
            if self.head_resume:
                f_in.writerow(fieldnames)
                self.head_resume = False
            f_in.writerow(data)
    
        
    def stack_states(self, action):
        i = 0
        immediate_reward = 0
        raw_states = []
        while i < self.agent.history_lengh:
            # Observe
            state = self.pre.processor(self.ale.getScreenGrayscale())
            immediate_reward += self.ale.act(self.legal_actions[action])
            raw_states.append(state)
            i += 1
        #reward = np.clip(immediate_reward, -1, 1)
        return immediate_reward, np.dstack(raw_states)

    def step_frame_stack(self):
        
        # Select one action
        action, type_action = self.agent.choose_action('train', np.expand_dims(np.divide(self.state, 255), axis=0))

        # Act and colect the reward and Observe the new state
        self.immediate_reward, self.next_state = self.stack_states(action)

        # Observe if is game over
        done = self.ale.game_over()

        # Store a new experience (s,a,r,s_1,d)
        self.agent.store_experience(np.asarray(self.state), action, self.immediate_reward, np.asarray(self.next_state), done)

        # Observe
        self.state = self.next_state
        
        return self.immediate_reward

    def show_image_of(self, state):
        image = tf.squeeze(state/255)
        plt.figure(figsize=(84, 84))
        plt.imshow(image, cmap=plt.cm.binary)
        plt.show()

    def step(self):

        # Observe
        state = self.pre.processor(self.ale.getScreenGrayscale())
        
        # Select one action
        action, type_action = self.agent.choose_action('train', np.expand_dims(np.divide(state, 255), axis=0))

        # Act and colect the reward
        self.immediate_reward = self.ale.act(self.legal_actions[action])
        #immediate_reward = np.clip(self.ale.act(self.legal_actions[action]), -1, 1)

        # Observe the new state
        next_state = self.pre.processor(self.ale.getScreenGrayscale())

        # Observe if is game over
        done = self.ale.game_over()

        # Store a new experience (s,a,r,s_1,d)
        self.agent.store_experience(np.asarray(state), action, self.immediate_reward, np.asarray(next_state), done)

        # Return the immediate reward
        return self.immediate_reward

    def save_checkpoints_of_train(self):
        self.checkpoint = self.flag_w

        if self.ale.getFrameNumber() >= 5000000 and self.flag_w == 0:
            self.agent.save_model_weights(self.folder_train, self.flag_w)
            print(" Save checkpoint of train : " + str(datetime.datetime.now()))
            self.flag_w = 5000000
        elif self.ale.getFrameNumber() >= 10000000 and self.flag_w == 5000000:
            self.agent.save_model_weights(self.folder_train, self.flag_w)
            print(" Save checkpoint of train : " + str(datetime.datetime.now()))
            self.flag_w = 10000000
        elif self.ale.getFrameNumber() >= 20000000 and self.flag_w == 10000000:
            self.agent.save_model_weights(self.folder_train, self.flag_w)
            print(" Save checkpoint of train : " + str(datetime.datetime.now()))
            self.flag_w = 20000000
        elif self.ale.getFrameNumber() >= 30000000 and self.flag_w == 20000000:
            self.agent.save_model_weights(self.folder_train, self.flag_w)
            print(" Save checkpoint of train : " + str(datetime.datetime.now()))
            self.flag_w = 30000000
        elif self.ale.getFrameNumber() >= 50000000 and self.flag_w == 30000000:
            self.agent.save_model_weights(self.folder_train, self.flag_w)
            print(" Save checkpoint of train : " + str(datetime.datetime.now()))
            self.flag_w = 50000000
        elif self.ale.getFrameNumber() >= 70000000 and self.flag_w == 50000000:
            self.agent.save_model_weights(self.folder_train, self.flag_w)
            print(" Save checkpoint of train : " + str(datetime.datetime.now()))
            self.flag_w = 70000000
        elif self.ale.getFrameNumber() >= 100000000 and self.flag_w == 70000000:
            self.agent.save_model_weights(self.folder_train, self.flag_w)
            print(" Save checkpoint of train : " + str(datetime.datetime.now()))
            self.flag_w = 100000000
        


    def train(self):

        # Load configs to train with GPU
        self.set_hardware_to_train()

        # Create a preprocessor of states of ALE
        self.pre = State_Preprocessor(self.WIDTH, self.HEIGHT)
        
        # Load configurations of ALE and ROM
        self.set_enviroment_configurations()
        
        # Create an A.I of RL
        self.agent = Agent_DQN(self.num_actions, self.input_dims)
        self.agent.update_weights_q_target()

        # Folder to save logs and other important things
        self.create_folder()

        # Flag to save weights
        self.flag_w = 0
        self.checkpoint = 0

        # Preper the train for many and many frames ...
        self.episode = 0
        self.step_env = 0

        # Loss
        self.loss = 0

        while (self.ale.getFrameNumber() < self.MAX_FRAMES):

            # Variables to manipulate loss in resume
            self.total_loss = .0
            self.count_step_loss = 0
            
            # Prepar game to start
            self.episode += 1
            self.episode_reward = 0
            self.ale.reset_game()
            
            self.immediate_reward = 0
            # initiate variables of state and next_state
            noop = 0
            _,self.state = self.stack_states(noop)

            self.action_step = 0 
            while not self.ale.game_over():
                # Interact of enviroment
                #self.episode_reward += self.step()
                self.episode_reward += self.step_frame_stack()
                
                if self.agent.memory_replay.lenght() > self.agent.start_train:
                    self.action_step += 1
                    self.step_env += 1

                    if self.action_step % self.agent.update_frequency == 0:
                        minibatch = self.agent.memory_replay.sample_minibatch()
                        # bat = np.asarray(minibatch)
                        # states = np.stack(np.divide(bat[...,0],255))
                        # for s in states:
                        #     self.show_image_of(s)
                        self.loss = float(self.agent.train_agent(minibatch, self.method))
                        self.read_data()
                        self.total_loss += self.loss
                        self.count_step_loss += 1
                        self.action_step = 0

                    if self.step_env % self.agent.target_update_frequency == 0:
                        
                        self.step_env = 0
                        self.agent.update_weights_q_target()
                        self.agent.save_model_weights(
                            self.folder_train, '_last')
                        print(" Copy Weights Q to Q' ")
                    
                    self.save_checkpoints_of_train()

            self.write_log_train()
            self.resume_episode()

    
    def teste_skip_frame(self):
       
        # Load configurations of ALE and ROM
        # ALE enviroment parameters
        self.ale.setInt(b'random_seed', 123)
        self.ale.setInt(b'frame_skip', 5)
        self.ale.setFloat(b'repeat_action_probability', .25)
        self.ale.setBool(b'color_averaging', True)
        self.ale.setBool(b'display_screen', False)
        
        # ROM Configs
        self.rom_path = 'rom/' + self.rom_name + '.bin'
        self.ale.loadROM(str.encode(self.rom_path))

        # Game mode
        self.ale.setMode(0)
        self.ale.setDifficulty(0)

        # Get the list of legal actions
        self.legal_actions = self.ale.getLegalActionSet()
        self.num_actions = len(self.ale.getLegalActionSet())

        while (self.ale.getFrameNumber() < self.MAX_FRAMES):
            while not self.ale.game_over():
                a = self.legal_actions[randrange(len(self.legal_actions))]
                reward = self.ale.act(a)
                state = self.ale.getScreenGrayscale()
                print('Total frames: {} Screen_shape: {}'.format(self.ale.getFrameNumber(),state.shape))

    def play(self):
        
        self.train_step = 1
        # Define the method ['dqn','ddqn']
        self.method = 'dqn'               
        # Define if you want train, evaluation or full evaluation
        self.mode = 'train'        
        # Define which one rom you can play ['space_invaders','beam_rider','breakout']
        self.rom_name = 'space_invaders'
        # Define the weights checkpoint of network Q  ['300000','500000','1000000','5000000'] (This numbers represent the amount of total frames)
        #self.checkpoint = '_last'
        # Define if you want see the screen when train or evaluation
        self.display_screen = False

        self.experiment = 'experiments_stack/'
        #self.teste_skip_frame()
        self.train()
        

env = Enviroment()
env.play()
