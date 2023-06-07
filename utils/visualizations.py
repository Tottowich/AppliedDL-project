import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from mpl_toolkits.axes_grid1 import ImageGrid

from data.data_generator import DataLoader
from data.pre_processor import pre_prosses
from utils.helper import prediction_threshold

def plot_sample(gen_data: DataLoader,
                arrays:List[str],
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
    n_formats = len(arrays)-1
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
    first_format = arrays[0]
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
            ax.set_title(f"{arrays[col]}", fontsize=20)
        elif row == 1:
            # Overlay the target on the input image.
            ax.imshow(x[col,...,col], cmap='gray', vmin=0, vmax=1)
            ax.imshow(alphas_label, cmap='Reds', vmin=0, vmax=1, alpha=alphas_label)
            ax.set_title(f"{arrays[-1]}", fontsize=20)
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
from uncertainty.monte_carlo_dropout import MonteCarloDropoutModel
def plot_predictions(gen,n:int=10, model=None, unc_model:MonteCarloDropoutModel=None, title:str=None,show_labels:bool=True):
    if not isinstance(gen, tuple):
        data_inputs, data_masks = next(iter(gen))
    else:
        data_inputs, data_masks = gen
    indexes = np.random.randint(0, len(data_inputs), n)
    rows = np.sqrt(n).astype(int)
    cols = np.ceil(n/rows).astype(int)
    fig = plt.figure(figsize=(20, 20), dpi=100)
    per_input = (1 + (model is not None) + (unc_model is not None))
    cols = per_input if per_input > 1 else cols
    grid = ImageGrid(fig, 111, nrows_ncols=(rows*per_input, cols), axes_pad=0.0)
    first_row = True
    for i, idx in zip(range(0,len(grid),per_input), indexes):
        x = data_inputs[idx, :, :]
        y = data_masks[idx, :, :]
        alphas_label = np.ma.masked_array(y, mask=y>=0)
        alphas_label = np.where(alphas_label, 0.8, 0).astype(np.float32).squeeze()
        grid[i].imshow(x[...,-2], cmap='gray', vmin=0, vmax=1)
        if show_labels:
            grid[i].imshow(alphas_label, cmap='Reds', vmin=0, vmax=1, alpha=alphas_label)
        grid[i].axis('off')
        if model: # Plot predictions
            i = i+1
            y_pred = prediction_threshold(model.predict(x[np.newaxis, ...], verbose=0)[0], threshold=0.6)
            alphas_pred = np.ma.masked_array(y_pred, mask=y>=0)
            alphas_pred = np.where(alphas_pred, 0.8, 0).astype(np.float32).squeeze()
            grid[i].imshow(x[...,-2], cmap='gray', vmin=0, vmax=1)
            grid[i].imshow(alphas_pred, cmap='Blues', vmin=0, vmax=1, alpha=alphas_pred)
            grid[i].axis('off')
        if unc_model: # Plot uncertainty
            i = i+1
            unc = unc_model.predict(x[np.newaxis, ...],n_iter=25)[0]
            alphas_unc = np.array(unc).squeeze()/np.max(unc)# np.ma.masked_array(unc, mask=unc>=1e-2)
            grid[i].imshow(x[...,-2], cmap='gray', vmin=0, vmax=1)
            grid[i].imshow(alphas_unc, cmap='Greens', vmin=0, vmax=1, alpha=alphas_unc)
            grid[i].axis('off')
    if model or unc_model:
        # Increase spacing between grid elements
        if first_row:
            grid[0].set_title('Input'+f"{' w/ Labels' if show_labels else ''}", fontsize=20)
            if model:
                grid[1].set_title('Prediction', fontsize=20)
            if unc_model:
                grid[2].set_title('Uncertainty', fontsize=20)
            first_row = False
    # if title:
    #     fig.suptitle(title, fontsize=20)
    plt.show()   