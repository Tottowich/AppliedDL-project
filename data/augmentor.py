# This file is for data augmentation of the training data for autoencoders, image reconstruction, and segmentation.
# Path: data/augmentor.py
# Compare this snippet from train_autoencoder.py:
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import AffineTransform, rescale, resize, rotate, warp
# Augmentation class for used in the training pipeline.
def prob(p: float) -> bool:
    return np.random.random() < p

class Augmentation:
    verbose: bool = False
    # Parent class for all augmentations
    def __init__(self, p: float):
        self.p = p
    @property
    def name(self) -> str:
        return self.__class__.__name__
    def __call__(self, x: np.ndarray,y: np.ndarray) -> np.ndarray:
        if prob(self.p):
            if self.verbose:
                print(f"Augmenting: Applying {self.name}")
            return self.augment(x,y)
        else:
            return x,y
    def augment(self, x: np.ndarray,y:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    

class Flip(Augmentation):
    def __init__(self, p: float = 0.5, axis: int = 0):
        super().__init__(p)
        self.axis = axis # 1 for horizontal, 0 for vertical
    def augment(self, x: np.ndarray,y:np.ndarray) -> np.ndarray:
        return np.flip(x, axis=self.axis),np.flip(y, axis=self.axis)

class Rotate(Augmentation):
    def __init__(self, p: float = 0.5, angle: float = np.pi/4):
        super().__init__(p)
        self.angle = angle
    def augment(self, x: np.ndarray,y:np.ndarray) -> np.ndarray:
        # Rotate image and fill with zeros
        # Random angle between -angle and angle
        angle = np.random.uniform(-self.angle, self.angle)
        # Batched rotate
        return np.array([rotate(img, angle, resize=False, mode="constant", cval=0) for img in x]),np.array([rotate(img, angle, resize=False, mode="constant", cval=0) for img in y])
class Noise(Augmentation):
    def __init__(self, p: float = 0.5, mean: float = 0.0, std: float = 0.1):
        super().__init__(p)
        self.mean = mean
        self.std = std
    def augment(self, x: np.ndarray,y:np.ndarray) -> np.ndarray:
        noise = np.random.normal(self.mean, self.std, x.shape)
        return (x + noise, y)

class Mask(Augmentation):
    def __init__(self, p: float = 0.5, max_n_masks: int = 10, mask_size: float = 0.5):
        super().__init__(p)
        self.max_n_masks = max_n_masks
        self.mask_size = mask_size
    def augment(self, x: np.ndarray, y:np.ndarray) -> np.ndarray:
        # Random number of masks
        n_masks = np.random.randint(1, self.max_n_masks)
        w = x.shape[-3]
        h = x.shape[-2]
        for _ in range(n_masks):
            # Random mask size
            mask_size = np.random.uniform(low=self.mask_size/2, high=self.mask_size)
            # Random mask position
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            x2 = int(x1 + w * mask_size)
            y2 = int(y1 + h * mask_size)
            x[:, x1:x2, y1:y2,:] = 0
        return (x, y)

class Translate(Augmentation):
    def __init__(self, p: float = 0.5, factor: float = 0.5):
        super().__init__(p)
        self.factor = factor
    def augment(self, x: np.ndarray,y:np.ndarray) -> np.ndarray:
        # Random translation factor
        tx = np.random.uniform(-self.factor, self.factor) * x.shape[0]
        ty = np.random.uniform(-self.factor, self.factor) * x.shape[1]
        # Affine transform, grayscale image so no need to transform channels
        tform = AffineTransform(translation=(tx, ty))
        # Apply transform, Image will be filled with zeros
        x = np.array([warp(img, tform.inverse, mode="constant", cval=0) for img in x])
        y = np.array([warp(img, tform.inverse, mode="constant", cval=0) for img in y])
        return x,y

class Shear(Augmentation):
    def __init__(self, p: float = 0.5, factor: float = 0.5):
        super().__init__(p)
        self.factor = factor
    def augment(self, x: np.ndarray,y:np.ndarray) -> np.ndarray:
        # Random shear factor
        shear_factor = np.random.uniform(-self.factor, self.factor)
        # Create affine transform
        tform = AffineTransform(shear=shear_factor)
        # Use warp to apply transform
        x = np.array([warp(img, tform.inverse, mode="constant", cval=0) for img in x])
        y = np.array([warp(img, tform.inverse, mode="constant", cval=0) for img in y])
        return x,y
class Augmentor:
    """
    Augmentations:
    flip_x: float
        Probability of flipping the image horizontally
    flip_y: float
        Probability of flipping the image vertically
    rotate: float
        Probability of rotating the image
    radians: float
        Maximum rotation angle in radians
    translate: float
        Probability of translating the image
    noise: float
        Probability of adding noise to the image
    noise_std: float
        Standard deviation of the noise
    noise_mean: float
        Mean of the noise
    mask: float
        Probability of masking the image
    max_n_masks: int
        Maximum number of masks
    mask_size: float
        Maximum size of the mask as a fraction of the image size
    shear: float
        Probability of shearing the image
    shear_factor: float
        Maximum shear factor
    """
    def __init__(self,
                flip_x:float=0.25,
                flip_y:float=0.25,
                rotate:float=0.5,
                radians:float=np.pi/6,
                translate:float=0.2,
                noise:float=0.25,
                noise_std:float=0.1,
                noise_mean:float=0.1,
                mask:float=0.8,
                max_n_masks:int=10,
                mask_size:float=0.25,
                shear:float=0.1,
                shear_factor:float=0.4,
                verbose:bool=False
                ):
        self.verbose = verbose
        self._active = True
        Augmentation.verbose = self.verbose
        self.augmentations = {}
        if noise > 0:
            self.augmentations["noise"] = Noise(p=noise, std=noise_std, mean=noise_mean)
        if flip_x > 0:
            self.augmentations["flip_x"] = Flip(p=flip_x, axis=1)
        if flip_y > 0:
            self.augmentations["flip_y"] = Flip(p=flip_y, axis=0)
        if rotate > 0:
            self.augmentations["rotate"] = Rotate(p=rotate, angle=radians)
        if translate > 0:
            self.augmentations["translate"] = Translate(p=translate, factor=translate)
        if mask > 0:
            self.augmentations["mask"] = Mask(p=mask, max_n_masks=max_n_masks, mask_size=mask_size)
        if shear > 0:
            self.augmentations.append(Shear(p=shear, factor=shear_factor))
    def __call__(self, x: np.ndarray,y:np.ndarray) -> np.ndarray:
        if self._active:
            # if x.shape[:-1] != y.shape:
            #     raise Exception("x and y must have the same shape")
            if len(x.shape) < 4:
                x = x[np.newaxis,...]
                y = y[np.newaxis,...]
            for aug in self.augmentations.values():
                x,y = aug(x,y)
        return x, y
    @property
    def keys(self):
        return list(self.augmentations.keys())
    @property
    def active(self):
        return self._active
    def scale_probability(self, key:str, factor:float):
        if self.verbose:
            print(f"Scaling probability of {key} by {factor:3.3e}: {self.augmentations[key].p:3.3e} -> {self.augmentations[key].p * factor:3.3e}")
        self.augmentations[key].p *= factor
    def set_active(self, active:bool):
        self._active = active
    def __repr__(self):
        return f"Augmentor({', '.join([f'{k}: {v.p:3.3e}' for k,v in self.augmentations.items()])})"