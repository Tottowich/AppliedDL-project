# TensorFlow basic imports
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.models import Model, Sequential
# Model specific imports
from keras.layers import Layer
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Activation, Dropout, Dense,Flatten, SpatialDropout2D, UpSampling2D
# Attention imports
from keras.layers import Attention, MultiHeadAttention
# Other imports
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
class NestedLayer(Sequential):
    """Nested layer"""
    def __init__(self, **kwargs):
        super(NestedLayer, self).__init__(**kwargs)
    def get_config(self):
        config = super(NestedLayer, self).get_config()
        return config

class ResidualConcatenation(Model):
    """Residual concatenation wrapper layer"""
    def __init__(self, fn,**kwargs):
        super(ResidualConcatenation, self).__init__(**kwargs)
        self.concat = keras.layers.Concatenate()
        self.fn = fn
    def call(self, x, **kwargs):
        return self.concat([x, self.fn(x)])
    def get_config(self):
        config = super(ResidualConcatenation, self).get_config()
        config.update({'fn': self.fn})
        return config

class ResidualLinearBlock(Layer):
    """Residual linear block"""
    def __init__(self, filters:int, kernel_size:int=3, strides:int=1, padding:str="same", activation:str="relu", **kwargs):
        super(ResidualLinearBlock, self).__init__(**kwargs)
        self.seq = Sequential([
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=False),
            BatchNormalization(),
            Activation(activation),
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=False),
            BatchNormalization(),
        ])
        self.activation = Activation(activation)
    def call(self, x, **kwargs):
        residual = x
        return self.activation(self.seq(x)+residual)
    def get_config(self):
        config = super(ResidualLinearBlock, self).get_config()
        config.update({'seq': self.seq})
        return config

# Custom Residual block with convolutional skip connection
class ResidualConvBlock(Layer):
    """Residual convolution block"""
    def __init__(self, filters:int, kernel_size:int=3, strides:int=1, padding:str="same", activation:str="relu", **kwargs):
        super(ResidualConvBlock, self).__init__(**kwargs)
        self.seq = Sequential([
            ConvBlock(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation),
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False),
            BatchNormalization(),
        ])
        self.skip = Sequential([
            Conv2D(filters=filters, kernel_size=1, strides=strides, padding=padding, use_bias=False),
            BatchNormalization()
        ])
        self.activation = Activation(activation)
       
    def call(self, x, **kwargs):
        residual = x
        y = self.activation(self.seq(x)+self.skip(residual))
        return y



class ConvBlock(NestedLayer):
    """Convolution block"""
    def __init__(self, filters:int, kernel_size:int=3, strides:int=1, padding:str="same", activation:str="relu", **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=False))
        self.add(BatchNormalization())
        self.add(Activation(activation))
    def get_config(self):
        return super().get_config()

class ResidualBlock(Sequential):
    def __init__(self,
                filters:int,
                kernel_size:int=3,
                strides:int=1,
                padding:str="same",
                activation:str="relu",
                depth:int=1,
                drop_rate:float=0.0, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.blocks = []
        for i in range(depth):
            if i==0:
                self.add(ResidualConvBlock(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation))
            else:
                self.add(ResidualLinearBlock(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation))
            if drop_rate>0.0:
                self.add(SpatialDropout2D(drop_rate))
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({'blocks': self.blocks})
        return config
    

