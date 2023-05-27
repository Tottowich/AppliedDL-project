import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

from data.data_generator import DataGenerator
from utils.visualizations import plot_sample


class PlotSampleCallback(Callback):
    """Callback that plots and saves sample predictions during training.
    """
    def __init__(self, gen_data:DataGenerator, save_path:str, threshold:float=0.5, title:str="Samples", **kwargs):
        super().__init__(**kwargs)
        self.gen_data = gen_data
        self.array_labels = gen_data.array_labels
        self.save_path = save_path
        self.threshold = threshold
        self.title = title
    def _plot(self, title, name):
        plot_sample(self.gen_data, self.array_labels, model=self.model, threshold=self.threshold, save=True, title=title, save_path=self.save_path, figure_name=name)
        
    def on_epoch_end(self, epoch, logs=None):
        title = f"{self.title}: epoch {epoch}"
        name = f"epoch_{epoch}.png"
        self._plot(title, name)
        
    def on_train_end(self, logs=None):
        title = f"{self.title}: final epoch"
        name = f"epoch_final.png"
        self._plot(title, name)
        
    def on_train_begin(self, logs=None):
        title = f"{self.title}: initial epoch"
        name = f"epoch_initial.png"
        self._plot(title, name)
        

class PlotMetricsCallback(Callback):
    """Plot loss during training along validation loss
    Based on code from: https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5
    """
    def __init__(self, epochs, n_batches,**kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.n_batches = n_batches
        self.dice_training = []
        self.dice_validation = []
        self.unc_validation = []
        self.dice_avg_traning = []
    def on_batch_end(self, batch, logs=None):
        out = super().on_batch_end(batch, logs)
        self.dice_training.append(logs.get('dice_coef'))
        return out
    def on_epoch_end(self, epoch, logs=None):
        out = super().on_epoch_end(epoch, logs)
        self.dice_validation.append(logs.get('val_dice_coef'))
        self.unc_validation.append(logs.get('val_monte_carlo_uncertainty'))
        self.dice_avg_traning.append(np.mean(self.dice_training[-self.n_batches:]))
        clear_output(wait=True)
        fig, axs = plt.subplots(1, 2, figsize=(16,8))
        axs[0].plot(np.linspace(0, epoch, epoch + 1), self.dice_validation, label="val_dice")
        axs[0].plot(np.linspace(0, epoch, epoch + 1), self.dice_avg_traning, label="train_dice", color='C1')
        axs[0].plot(np.linspace(0, epoch, self.n_batches*(epoch+1)), self.dice_training, label="train_dice_batch", alpha=0.3, color='C1')
        axs[0].legend()
        axs[1].plot(np.linspace(0, epoch, epoch + 1), self.unc_validation, label="val_unc")
        axs[0].set_title("Dice score")
        axs[0].grid(True)
        axs[0].set_xlabel('Epoch')
        axs[1].set_title("MC Uncertainty")
        axs[1].grid(True)
        axs[1].set_xlabel('Epoch')
        plt.show()
        return out
            