import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential

class Mymodel(Model):
    
    def __init__(self,input_dim,output_dim):
        super(Mymodel,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.in_conv1   = layers.Conv2D(32, kernel_size=(8,8),strides=4, input_shape=self.input_dim, activation='relu', use_bias=False) 
        self.conv2      = layers.Conv2D(64, kernel_size=(4,4),strides=2, activation='relu', use_bias=False) 
        self.conv3      = layers.Conv2D(64, kernel_size=(3,3),strides=1, activation='relu', use_bias=False)
        self.flatten    = layers.Flatten()
        self.dense      = layers.Dense(512, activation='relu',use_bias=False) 
        self.out_dense  = layers.Dense(self.output_dim, activation=None)
        
    def _build_model(self):
        
        self.model = Sequential([
            self.in_conv1,
            self.conv2,      
            self.conv3,      
            self.flatten,    
            self.dense,      
            self.out_dense  
        ])
        return self.model