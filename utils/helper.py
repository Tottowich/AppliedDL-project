import os
import json
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import tensorflow as tf
def prediction_threshold(y_pred:np.ndarray,threshold:float=0.5):
    y_pred = np.where(y_pred > threshold, 1, 0)
    return y_pred

def create_dirs(model_name:str, exist_ok=True)->Tuple[str,str,str]:
    model_dir = os.path.join("models", model_name)
    if os.path.exists(model_dir) and not exist_ok:
        input(f"Model {model_name} already exists. Press enter to overwrite or ctrl+c to cancel.")
    os.makedirs(model_dir, exist_ok=exist_ok)
    figure_path = os.path.join(model_dir, "figures")
    os.makedirs(figure_path,exist_ok=exist_ok)
    checkpoint_path = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=exist_ok)
    return model_dir, figure_path, checkpoint_path
def write_setup(model_dir:str, setup:Dict[str,Union[str,int,float,bool]]):
    with open(os.path.join(model_dir, "setup.json"), "w") as f:
        json.dump(setup, f, indent=4)
        
def gpu_setup(fp16:bool=False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU(s) available (using '{gpus[0].name}'). Training will be lightning fast!")
        if fp16:
            details = tf.config.experimental.get_device_details(gpus[0])
            compute_capability = details.get('compute_capability')
            print("Compute capability:", compute_capability)
            if compute_capability[0] > 6:
                print("Turning on mixed_float16")
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
    else:
        print("No GPU(s) available. Training will be suuuuper slow!")