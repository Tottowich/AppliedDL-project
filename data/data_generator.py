
import os

import numpy as np
import tensorflow.keras as keras
from keras.utils import Sequence
from scipy.ndimage import zoom
from typing import List, Tuple, Dict, Union, Optional
from .augmentor import Augmentor

class DataLoader(Sequence):
    def __init__(self,
                 data_path,
                 arrays,
                 batch_size=32,
                 shuffle=True,
                 zoom_factor=0.25):
        self.data_path = data_path
        self.arrays = arrays
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.zoom_factor = zoom_factor

        if self.data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(self.data_path):
            raise ValueError('The data path is incorrectly defined.')

        self.file_list = [os.path.join(self.data_path, s) for s in os.listdir(self.data_path)]
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        with np.load(self.file_list[0]) as npzfile:
            self.in_dims = []
            self.n_channels = 1
            for i in range(len(self.arrays)):
                im = npzfile[self.arrays[i]]
                im = zoom(im, self.zoom_factor)
                self.in_dims.append((self.batch_size,
                                    *np.shape(im),
                                    self.n_channels))

        self.cached_data = {}  # Store preprocessed data and masks for each file ID

    def __len__(self):
        """Get the number of batches per epoch."""
        return max(int(np.floor(len(self.file_list) / self.batch_size)), 1)

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.file_list[i] for i in batch_indexes]

        inputs, masks = self.data_generator(batch_files)
        inputs, masks = self.post_process(inputs, masks)
        return inputs, masks
    def post_process(self, inputs, masks):
        return inputs, masks
    def on_epoch_end(self):
        """Update indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    def _process_raw(self, file):
        with np.load(file) as npzfile:
            input_images = []
            for array in self.arrays[:-1]:
                x = npzfile[array].astype(np.float32)
                if np.max(x) > 0:
                    x /= np.max(x)
                x = zoom(x, self.zoom_factor)
                x = np.expand_dims(x, axis=2)
                input_images.append(x)

            stacked_input = np.concatenate(input_images, axis=-1)
            input_data = stacked_input

            mask_raw = npzfile['mask'].astype(np.float32)
            mask_preprocessed = zoom(mask_raw, self.zoom_factor)
            mask_preprocessed = (mask_preprocessed > 0.5).astype(np.float32)
            mask_preprocessed = np.expand_dims(mask_preprocessed, axis=2)
            mask_data = mask_preprocessed
        self.cached_data[file] = (input_data, mask_data)
        return input_data, mask_data
    def data_generator(self, batch_files):
        inputs = []
        masks = []

        for file in batch_files:
            if file in self.cached_data:
                # Use cached data if available
                input_data, mask_data = self.cached_data[file]
            else:
                # Preprocess data and store in cache
                input_data, mask_data = self._process_raw(file)                    

            inputs.append(input_data)
            masks.append(mask_data)

        return np.array(inputs), np.array(masks)
class DataGeneratorAugmented(DataLoader):
    def __init__(self, *args, augmentor:Augmentor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentor = augmentor
        self._augment = self.augmentor is not None
    def set_augment(self, augment:bool):
        self._augment = augment
    def post_process(self, inputs, masks):
        if self._augment:
            inputs, masks = self.augmentor(inputs, masks)
        return inputs, masks
class DataGeneratorAutoEncoder(DataLoader):
    pass

def loaders(gen_dir:str,
            batch_size: int,
            augment:Union[bool,Augmentor] = True,
            array_labels: List[str] = ['t1', 't1ce', 't2', 'flair', 'mask'],
            fraction:float = 1.0,
            zoom_factor:float = 0.25):
    if isinstance(augment, bool):
        augmentor = Augmentor(translate=0, # No translation. Due to lack of speed.
                            shear=0, # No shear. Due to lack of speed.
                            rotate=0, # No rotation. Due to lack of speed.
                            mask=0.4, # Probability of masking the image.
                            mask_size=0.2, # Maximum size of the mask as a fraction of the image size.
                            max_n_masks=6, # Maximum number of masks to apply.
                            noise=0.2, # Probability of adding Gaussian noise to the image.
                            noise_mean=0.05, # Mean of the noise.
                            noise_std=0.1, # Standard deviation of the noise.
                            ) if augment else None# Augmentation of the data.
    else:
        assert isinstance(augment, Augmentor), f"\'augment\' must be a boolean or an Augmentor object. Not {type(augment)}."
    gen_train = DataGeneratorAugmented(data_path=gen_dir + 'training',
                            arrays=array_labels,
                            batch_size=batch_size,
                            augmentor=augmentor,
                            zoom_factor=zoom_factor)
    gen_val = DataGeneratorAugmented(data_path=gen_dir + 'validating',
                            arrays=array_labels,
                            batch_size=batch_size,
                            zoom_factor=zoom_factor)
    gen_test = DataGeneratorAugmented(data_path=gen_dir + 'testing',
                            arrays=array_labels,
                            batch_size=batch_size,
                            zoom_factor=zoom_factor)
    return gen_train, gen_val, gen_test, augmentor