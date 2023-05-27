# TensorFlow basic imports
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.models import Model, Sequential
# Model specific imports
from keras.layers import Layer
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Activation, Dropout, Dense,Flatten, SpatialDropout2D, LayerNormalization
# Attention imports
from keras.layers import Attention, MultiHeadAttention
# Other imports
import numpy as np
from typing import List, Tuple, Dict, Union, Optional

class PositionalEmbedding(Layer):
    def __init__(self, max_len:int=1000, embed_dim:int=768, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.position_embedding = self.add_weight(shape=(max_len, embed_dim), initializer="random_normal", trainable=True)
        self.class_embedding = self.add_weight(shape=(1, embed_dim), initializer="random_normal", trainable=True)

    def call(self, x):
        # Add the position embedding to the input
        batch_size = tf.shape(x)[0]
        num_patches = tf.shape(x)[1]
        # Add the class token to the input
        class_token = tf.broadcast_to(self.class_embedding, (batch_size, 1, self.embed_dim))
        x = tf.concat([class_token, x], axis=1)
        # Add the position embedding to the input
        position_embedding = tf.broadcast_to(self.position_embedding, (batch_size, num_patches+1, self.embed_dim))
        return x + position_embedding
class PatchEmbedding(Layer):
    # Patch/Position embedding from the paper: An image is worth 16x16 words: Transformers for image recognition at scale
    def __init__(self, patch_size:int=16, embed_dim:int=768, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        # Input image has size: (batch_size, height, width, channels)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Patch the image into patches using a convolution
        self.patch_conv = Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size)
        # Rearrange the patches into: (batch_size, num_patches, embed_dim)
        self.reshape = keras.layers.Reshape((-1, embed_dim))
        # Add a position embedding and a class token
        self.position_embedding = PositionalEmbedding()


    def call(self, x:tf.Tensor, **kwargs)->tf.Tensor:
        x = self.patch_conv(x)
        x = self.reshape(x)
        x = self.position_embedding(x)
        return x





class VisualAttention(Model):
    """Visual attention from the paper: An image is worth 16x16 words: Transformers for image recognition at scale"""
    def __init__(self, d_model:int, num_heads:int=8, patch_size:int=16,embed_dim:int=768,**kwargs):
        super(VisualAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Patch/Position embedding
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense = keras.layers.Dense(d_model)
        self.layernorm = LayerNormalization()
    def call(self, x:tf.Tensor, **kwargs)->tf.Tensor:
        x = self.patch_embedding(x)
        x = self.attention(x,x)
        x = self.dense(x)
        x = self.layernorm(x)
        return x
        
def test_build_model():
    # Test the model
    model = VisualAttention(d_model=512, num_heads=8, patch_size=16, embed_dim=768)
    model.build(input_shape=(None, 224, 224, 3))
    print(model.summary())


if __name__=="__main__":
    test_build_model()