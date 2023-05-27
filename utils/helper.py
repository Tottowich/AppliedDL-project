import os

import numpy as np
from typing import List, Tuple, Dict, Union, Optional

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
    
