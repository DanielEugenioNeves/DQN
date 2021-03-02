from collections import deque
import random 
import numpy as np
import os
class Buffer:
    def __init__(self,replay_memory_size,minibatch_size):
        self.minibatch_size = minibatch_size
        self.max_replay_memory_size = replay_memory_size
        self.buffer_experiences = deque(maxlen=self.max_replay_memory_size)

    def store(self,state,action,reward,next_state,terminated):
        experience = state,action,reward,next_state,terminated
        self.buffer_experiences.append(experience)

    def sample_minibatch(self):
        minibatch = random.sample(self.buffer_experiences,self.minibatch_size)
        return minibatch
    
    def lenght(self):
        return len(self.buffer_experiences)
        
    def save_on_folder(self,rom):
        """Save the replay buffer to a folder"""
        folder_name = 'experiences/' + rom
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        np.save(folder_name + '/buffer.npy', self.buffer_experiences)

    def load_on_folder(self,rom):
        """Loads the replay buffer from a folder"""
        folder_name = 'experiences/' + rom
        data = np.load(folder_name + '/buffer.npy',allow_pickle=True)
        for d in data:
            self.buffer_experiences.append(d)
        l = self.lenght()        