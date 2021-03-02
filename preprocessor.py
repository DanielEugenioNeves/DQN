import tensorflow as tf
import numpy as np
class State_Preprocessor:

    def __init__(self,width,height):
        self.width = width
        self.height = height
        pass
    #NEAREST_NEIGHBOR
    #MITCHELLCUBIC
    def processor(self,state):
        #self.state_processor = tf.image.rgb_to_grayscale(state)
        #self.state_processor = tf.image.crop_to_bounding_box(state,34,0,160,160)
        self.state_processor = tf.image.resize(state, (self.width, self.height), method=tf.image.ResizeMethod.MITCHELLCUBIC)
        return self.state_processor