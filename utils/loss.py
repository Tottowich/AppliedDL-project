import tensorflow.keras as keras
from keras import backend as K
from keras.losses import BinaryFocalCrossentropy


def dice_coef(y_true,y_pred, smooth=100):        
    
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return dice

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))

# Custom loss function for keras containing dice loss and binary focal loss
class FocalDiceLoss(keras.losses.Loss):
    def __init__(self, w_focal,w_dice,gamma=2.0, alpha=0.25, smooth=100, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        self.focal_loss = BinaryFocalCrossentropy(gamma=gamma, alpha=alpha)
        self.w_focal = w_focal
        self.w_dice = w_dice
    def call(self, y_true, y_pred):
        # Compute focal loss
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        dice_loss = dice_coef_loss(y_true, y_pred)
        focal_loss = self.focal_loss(y_true, y_pred)
        return self.w_dice*dice_loss + self.w_focal*focal_loss