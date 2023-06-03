# TensorFlow basic imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
# Model specific imports
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Activation, Dropout, Dense,Flatten, SpatialDropout2D, UpSampling2D
# Attention imports
from tensorflow.keras.layers import Attention, MultiHeadAttention
# Other imports
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
# Plot model
from tensorflow.keras.utils import plot_model
if __name__=="__main__":
    # Resolve relative imports
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    from model_utils import ResidualConcatenation, ResidualLinearBlock, ResidualConvBlock,ResidualBlock #ResidualAddition,
else:
    from .model_utils import ResidualConcatenation, ResidualLinearBlock, ResidualConvBlock,ResidualBlock #ResidualAddition,
# This file contains encoder classes for the segmentation models of brain tumor segmentation.
# The encoders are used to extract features from the input image, creating a low level representation of the image.
# The encoders are used in the U-Net architecture.

class EncoderBlock(Layer):
    # Parent class for residual, attention and other encoder blocks
    depth_id = 0
    def __init__(self, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        EncoderBlock.depth_id += 1
    def call(self, x, **kwargs):
        raise NotImplementedError
    def get_config(self):
        config = super(EncoderBlock, self).get_config()
        return config

# class ResidualEncoderBlock(EncoderBlock):
#     """Encoder block"""
#     def __init__(self,
#                 filters:int,
#                 kernel_size:int=3,
#                 strides:int=1,
#                 padding:str="same",
#                 activation:str="relu",
#                 depth:int=1,
#                 drop_rate:float=0.0, **kwargs):
#         super(ResidualEncoderBlock, self).__init__(name=f"residual_encoder_block_{EncoderBlock.depth_id}", **kwargs)
#         self.seq = Sequential()
#         for i in range(depth):
#             if i==0:
#                 self.seq.add(ResidualConvBlock(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation))
#             else:
#                 self.seq.add(ResidualLinearBlock(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation))
#             if drop_rate>0.0:
#                 self.seq.add(SpatialDropout2D(drop_rate))
#     def call(self, x, **kwargs):
#         return self.seq(x)
#     def get_config(self):
#         config = super(ResidualEncoderBlock, self).get_config()
#         config.update({'seq': self.seq})
#         return config
    # Custom repr

class ResidualEncoderBuilder:
    """Encoder"""
    def __init__(self,
                input_shape:Tuple[int, int, int],
                filters:List[int],
                kernel_size:int=3, 
                strides:int=1,
                padding:str="same", # Padding for the convolutional layers
                activation:str="relu", # Activation function
                depth:List[int]=[1], # Depth of each block
                drop_rate:float=0.0, **kwargs):
        self.seq = []
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.depth = depth
        self.drop_rate = drop_rate
        self._make_lists()
        # If depth is not as long as filters, repeat the last value
        for i in range(len(filters)):
            assert self.depth[i] > 0, "Depth must be greater than 0"
            s = Sequential(name=f"residual_encoder_block_{i}")
            s.add(ResidualBlock(filters=filters[i], 
                                kernel_size=self.kernel_size[i], 
                                strides=self.strides[i], 
                                padding=self.padding[i], 
                                activation=self.activation[i], 
                                depth=self.depth[i], 
                                drop_rate=self.drop_rate[i]))
            s.add(MaxPooling2D(pool_size=2, strides=2))

            self.seq.append(s)
        inp = Input(shape=input_shape)
        self.output = self._init_call(inp)
        # Get output shapes:
        self.output_shapes = [o.shape for o in self.output]
        self.model = Model(inputs=inp, outputs=self.output, name="Encoder")
    def _init_call(self, x, **kwargs):
        ys = []
        for i in range(len(self.seq)):
            x = self.seq[i](x)
            ys.append(x)
        for i in range(len(ys)//2):
            ys[i] = ys[i]
        return ys
    def initialize(self):
        pass
    def _make_lists(self):
        # If the parameters are not lists of same length, extend them with the last value
        if not isinstance(self.filters,list):
            self.filters = [self.filters]
        if not isinstance(self.kernel_size,list):
            self.kernel_size = [self.kernel_size]
        if not isinstance(self.strides,list):
            self.strides = [self.strides]
        if not isinstance(self.padding,list):
            self.padding = [self.padding]
        if not isinstance(self.activation,list):
            self.activation = [self.activation]
        if not isinstance(self.depth,list):
            self.depth = [self.depth]
        if not isinstance(self.drop_rate,list):
            self.drop_rate = [self.drop_rate]
        # Extend the lists to the same length, according to the number of filters
        n_layers = len(self.filters)
        self.kernel_size = self.kernel_size + [self.kernel_size[-1]]*(n_layers-len(self.kernel_size))
        self.strides = self.strides + [self.strides[-1]]*(n_layers-len(self.strides))
        self.padding = self.padding + [self.padding[-1]]*(n_layers-len(self.padding))
        self.activation = self.activation + [self.activation[-1]]*(n_layers-len(self.activation))
        self.depth = self.depth + [self.depth[-1]]*(n_layers-len(self.depth))
        self.drop_rate = self.drop_rate + [self.drop_rate[-1]]*(n_layers-len(self.drop_rate))
        # Check that the lists have the same length
        assert n_layers == len(self.kernel_size) == len(self.strides) == len(self.padding) == len(self.activation) == len(self.depth) == len(self.drop_rate), "The parameters must be lists of same length"
def build_encoder(input_shape:Tuple[int, int, int],
                filters:List[int],
                kernel_size:int=3, 
                strides:int=1,
                padding:str="same", # Padding for the convolutional layers
                activation:str="relu", # Activation function
                depth:List[int]=[1], # Depth of each block
                drop_rate:float=0.0, **kwargs):
    return ResidualEncoderBuilder(input_shape=input_shape,
                                  filters=filters, 
                                  kernel_size=kernel_size, 
                                  strides=strides, 
                                  padding=padding, 
                                  activation=activation, 
                                  depth=depth, 
                                  drop_rate=drop_rate).model
def initialize_encoder(input_shape:Tuple[int, int, int],
                filters:List[int],
                kernel_size:int=3, 
                strides:int=1,
                padding:str="same", # Padding for the convolutional layers
                activation:str="relu", # Activation function
                depth:List[int]=[1], # Depth of each block
                drop_rate:float=0.0, **kwargs):
    return ResidualEncoderBuilder(input_shape=input_shape,
                                  filters=filters, 
                                  kernel_size=kernel_size, 
                                  strides=strides, 
                                  padding=padding, 
                                  activation=activation, 
                                  depth=depth, 
                                  drop_rate=drop_rate).output

def test_blocks():
    H = 256
    W = 256
    C = 1
    x = Input(shape=(H,W,C))
    #x = tf.random.normal((1,H,W,C))
    encoder = build_encoder((H,W,C),filters=[2,4,8,16,32,64,128,256], kernel_size=3, strides=1, padding="same", activation="relu", depth=[3,2,1], drop_rate=0.0)
    ys = encoder(x)
    print(encoder.summary(expand_nested=True))
    plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True,expand_nested=True)

if __name__=="__main__":
    print("This is the encoder.py file.")
    test_blocks()
