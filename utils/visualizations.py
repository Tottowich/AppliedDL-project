import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from mpl_toolkits.axes_grid1 import ImageGrid

from data.data_generator import DataGenerator
from data.pre_processor import pre_prosses
from utils.helper import prediction_threshold

def plot_sample(gen_data: DataGenerator,
                array_labels:List[str],
                model:Model=None,
                batch_idx:int=None,
                threshold:float=0.5,
                save:bool=False,
                save_path:str="./figures/",
                figure_name:str="sample.png",
                title:str="Samples",):
    # Get a batch of data.
    idx = batch_idx if batch_idx is not None else np.random.randint(0,len(gen_data))
    data = gen_data[idx]
    x, y = pre_prosses(data)
    # Preprocess the data.
    # Grid with one of each array_label.
    n_formats = len(array_labels)-1
    rows = 2 if not model else 3
    cols = n_formats
    preds = model.predict(x, verbose=0) if model else None
    if threshold:
        preds = prediction_threshold(preds,threshold) if model else None
    # Create a figure with the correct number of subplots.
    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(rows, cols), 
                    axes_pad=[0.0, 0.35],
                    )
    first_format = array_labels[0]
    for i, ax in enumerate(grid):
        # Get the row and column index.
        # Create masked images.
        row = i // cols
        col = i % cols
        alphas_label = np.ma.masked_array(y[col], mask=y[col]>=0)
        alphas_label = np.where(alphas_label, 0.7, 0).astype(np.float32).squeeze()
        # Plot the data.
        if row == 0:
            # ax.imshow(x[i+col*n_formats], cmap='gray')
            ax.imshow(x[col,...,col], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"{array_labels[col]}", fontsize=20)
        elif row == 1:
            # Overlay the target on the input image.
            ax.imshow(x[col,...,col], cmap='gray', vmin=0, vmax=1)
            ax.imshow(alphas_label, cmap='Reds', vmin=0, vmax=1, alpha=alphas_label)
            ax.set_title(f"{array_labels[-1]}", fontsize=20)
        else:
            alphas = np.ma.masked_array(preds[col], mask=preds[col]>=0)
            # To binary mask
            alphas_pred = np.where(alphas, 0.8, 0).astype(np.float32).squeeze()
            ax.imshow(x[col,...,col], cmap='gray', vmin=0, vmax=1)
            ax.imshow(alphas_label, cmap='Reds', vmin=0, vmax=1, alpha=alphas_label)
            ax.imshow(alphas_pred, cmap='Blues', vmin=0, vmax=1, alpha=alphas_pred)
            ax.set_title(f"Prediction", fontsize=20)
            if row==0 and col==0:
                ax.legend(["Input", "Target", "Prediction"], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=20)

        ax.axis('off')
    fig.suptitle(title, fontsize=30)
    # Tight 
    # plt.tight_layout()
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(os.path.join(save_path,figure_name))
        plt.close(fig)
    else:
        plt.show()