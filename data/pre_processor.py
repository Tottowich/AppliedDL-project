import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from data.augmentor import Augmentor
class PreProcessor:
    def __init__(self, W:int,H:int,C:int,batch_size:int, per_batch:int, augmentor:Augmentor):
        self.W = W
        self.H = H
        self.C = C
        self.per_batch = per_batch
        self.augmentor = augmentor
        self.index = np.tile(np.arange(batch_size),(1,per_batch)).reshape(-1)
    def __call__(self, x:List[np.ndarray],augment:bool=True):
        y = x[-1]
        x = x[:-1]
        x = np.array(x)
        # Shape: (batch_size, per_batch, height, width, channels)
        # Reshape to: (batch_size*per_batch, height, width, channels)
        x = x.reshape((-1,self.W,self.H,self.C))
        # Repeat y per_batch times to match the shape of x.
        # y = np.repeat(y[:,np.newaxis],per_batch, axis=1)
        # Reshape to: (batch_size*per_batch, height, width, channels) with repeatition of y
        y = y[self.index]
        # Augment the data.
        if augment:
            x, y = self.augmentor(x,y)
        return x, y

def pre_prosses(data:List[np.ndarray]):
    inputs = data[:-1]
    targets = data[-1]
    inputs = np.stack(inputs, axis=-1).squeeze()
    return inputs, targets