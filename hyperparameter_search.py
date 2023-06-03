import os
from typing import List, Tuple, Dict, Union, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
# random.seed(2023)

import numpy as np

# np.random.seed(2023)  # Set seed for reproducibility
import keras
import tensorflow as tf
from tqdm import tqdm

# tf.random.set_seed(2023)
tf.config.run_functions_eagerly(True)
keras.backend.clear_session()
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import keras_tuner as kt
import wandb
name = "tumor-segmentation-keras-tuner"
wandb.init(project=name)
from wandb.keras import WandbCallback


from archs.segmentation.unet import build_unet
from data.data_generator import loaders
from utils.helper import create_dirs, write_setup, gpu_setup
from utils.loss import FocalDiceLoss, dice_coef, dice_coef_loss
from utils.visualizations import plot_sample
from archs.hpo import hyperparameter_build
gpu_setup(fp16=False)

gen_dir = "./data/"
model_dir = "./search/"
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
max_iter = 40
batch_size = 64


tuner = kt.BayesianOptimization(
    hyperparameter_build,
    objective='val_loss',
    # max_epochs=max_iter,
    # factor=3,
    max_trials=60,
    directory=model_dir,
    project_name=name,
    overwrite=True,
    seed=2023,
    logger=wandb,
)

gen_train, gen_val, gen_test, _ = loaders(gen_dir=gen_dir, batch_size=batch_size, augment=True, array_labels=array_labels)
tuner.search(gen_train, epochs=max_iter, validation_data=gen_val, callbacks=[WandbCallback(), early_stopping, reduce_lr])
# tuner.search(gen_train, epochs=max_iter, validation_data=gen_val, callbacks=[early_stopping, reduce_lr])