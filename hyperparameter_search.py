import os
from typing import List, Tuple, Dict, Union, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
# random.seed(2023)
import argparse
from argparse import BooleanOptionalAction
parser = argparse.ArgumentParser()
# FP16 default is False
parser.add_argument("--fp16", action=BooleanOptionalAction, help="Use mixed precision FP16", default=False)
parser.add_argument("--wandb", action=BooleanOptionalAction, help="Use wandb", default=False)
parser.add_argument("--neptune", action=BooleanOptionalAction, help="Use neptune", default=False)
parser.add_argument("--tensorboard", action=BooleanOptionalAction, help="Use tensorboard", default=False)
args = parser.parse_args()
use_fp16 = args.fp16
use_wandb = args.wandb
use_neptune = args.neptune
use_tensorboard = args.tensorboard
import numpy as np

# np.random.seed(2023)  # Set seed for reproducibility
import keras
import tensorflow as tf
from tqdm import tqdm

# tf.random.set_seed(2023)
tf.config.run_functions_eagerly(True)
keras.backend.clear_session()
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, \
    TensorBoard
from keras.optimizers import Adam
import keras_tuner as kt

from archs.segmentation.unet import build_unet
from data.data_generator import loaders
from utils.helper import create_dirs, write_setup, gpu_setup
from utils.loss import FocalDiceLoss, dice_coef, dice_coef_loss
from utils.visualizations import plot_sample
# from archs.hpo import hyperparameter_build
gpu_setup(fp16=use_fp16)
# Set up directories and variables
array_labels = ['t1', 't1ce', 't2', 'flair', 'mask']
name = "tumor-segmentation-keras-tuner"
gen_dir = "/home/thjo/Datasets/Brats/"
model_dir = "./search/"
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
max_iter = 40
batch_size = 64
# Initialize WandB if specified
if use_wandb:
    print(f"Initalizing wandb for project {name}")
    import wandb
    from wandb.keras import WandbCallback
    wandb.init(project=name)
elif use_neptune:
    print(f"Initalizing neptune for project {name}")
    import neptune.legacy as neptune
    import neptunecontrib.monitoring.kerastuner as npt_utils
    neptune.init(project_qualified_name=name)
    neptune.create_experiment(name=name, params={"fp16": use_fp16})
    neptune.create_experiment('bayesian-sweep')
elif use_tensorboard:
    log_dir = os.path.join(model_dir, "logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

# Create the tuner


input_shape = (64, 64, len(array_labels) - 1)
num_classes = 1
max_depth = 6
kernel_size = 3
strides = 1
activations = ["relu", "gelu", "selu"]
padding  = "same"
output_activation = "sigmoid"
decoder_types = ["concat", "add"]
upsample_types = ["transposed", "bilinear"]
losses = {"focal": FocalDiceLoss, "dice": dice_coef_loss}
losses_keys = list(losses.keys())
filter_step = 16
depth = 5
def hyperparameter_build(kt:kt.HyperParameters):
    depth = kt.Int('depth', min_value=4, max_value=max_depth, step=1)
    filters = [kt.Int(f'filter_{i}', min_value=2**(i+3), max_value=2**(i+5), step=filter_step) for i in range(depth)]
    activation = kt.Choice('activation', values=activations)
    depth_encoder = [kt.Int(f'depth_encoder_{i}', min_value=1, max_value=i+1, step=1) for i in range(depth)]
    depth_decoder = [kt.Int(f'depth_decoder_{i}', min_value=1, max_value=i+1, step=1) for i in range(depth)]
    drop_rate_encoder = [kt.Float(f'drop_rate_encoder_{i}', min_value=0.0, max_value=0.3, step=0.1, default=0.05) for i in range(depth)]
    drop_rate_decoder = [kt.Float(f'drop_rate_decoder_{i}', min_value=0.0, max_value=0.3, step=0.1, default=0.05) for i in range(depth)]
    output_depth = kt.Int('output_depth', min_value=1, max_value=6, step=1)
    decoder_type = kt.Choice('decoder_type', values=decoder_types)
    upsample_type = kt.Choice('upsample_type', values=upsample_types)
    unet = build_unet(
            input_shape=input_shape,
            num_classes=num_classes,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            depth_encoder=depth_encoder,
            depth_decoder=depth_decoder,
            drop_rate_encoder=drop_rate_encoder,
            drop_rate_decoder=drop_rate_decoder,
            output_depth=output_depth,
            output_activation=output_activation,
            decoder_type=decoder_type,
            upsample_type=upsample_type,
        )
    loss = losses[kt.Choice('loss', values=losses_keys)]
    if loss == FocalDiceLoss:
        gamma = kt.Float('gamma', min_value=0.0, max_value=5.0, step=0.25)
        w_focal = kt.Float('w_focal', min_value=0.0, max_value=1, step=0.05)
        w_dice = 1 - w_focal
        loss = loss(w_focal=w_focal,w_dice=w_dice,gamma=gamma)
    learning_rate = kt.Float('learning_rate', min_value=5e-4, max_value=5e-2, default=1e-2, step=1e-3)
    weight_decay = kt.Float('weight_decay', min_value=1e-6, max_value=5e-2, default=1e-3, step=1e-3)
    optimizer = Adam(learning_rate=learning_rate, decay=weight_decay)
    unet.compile(optimizer=optimizer, loss=loss, metrics=[dice_coef])
    return unet
if use_wandb:
    logger = WandbCallback()
elif use_neptune:
    logger = npt_utils.NeptuneLogger()
else:
    logger = None
tuner = kt.BayesianOptimization(
    hyperparameter_build,
    objective=kt.Objective("val_dice_coef", direction="max"),
    max_trials=60,
    directory=model_dir,
    project_name=name,
    overwrite=True,
    seed=2023,
    logger=logger,
)

callbacks = []
if use_wandb:
    callbacks.append(WandbCallback())
elif use_neptune:
    callbacks.append(npt_utils.NeptuneCallback(log_models=True))
elif use_tensorboard:
    callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min')
callbacks.append(early_stopping)
callbacks.append(reduce_lr)

gen_train, gen_val, gen_test, _ = loaders(gen_dir=gen_dir, batch_size=batch_size, augment=True, array_labels=array_labels)
tuner.search(gen_train, epochs=max_iter, validation_data=gen_val, callbacks=callbacks)

tuner.results_summary()