from archs.segmentation.encoder import build_encoder#, initialize_encoder
from archs.segmentation.decoders import build_decoder#, initialize_decoder
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from keras.layers import Input
from keras.models import Model
# Plot model
# from tensorflow.keras.utils import plot_model
from typing import List, Tuple, Dict, Union, Optional


def build_unet(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    filters: List[int],
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]],
    padding: str,
    activation: str,
    depth_encoder: List[int],
    decoder_type: str, # "concat", "add", "noskip"
    upsample_type: str, # "transposed", "bilinear", "nearest", "cubic"
    depth_decoder: List[int],
    drop_rate_encoder: List[float],
    drop_rate_decoder: List[float],
    output_depth: int,
    output_activation: str,
):
    encoder = build_encoder(
        input_shape=input_shape,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        depth=depth_encoder,
        drop_rate=drop_rate_encoder,
    )
    x = Input(shape=input_shape)
    encoder_outputs = encoder(x)
    decoder = build_decoder(
        encoder_outputs=encoder_outputs,
        num_classes=num_classes,
        filters=filters[::-1][1:],
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        depth=depth_decoder,
        output_depth=output_depth,
        output_activation=output_activation,
        drop_rate=drop_rate_decoder,
        decoder_type=decoder_type,
        upsample_type=upsample_type,
    )
    y = decoder(encoder_outputs[::-1])
    unet = Model(inputs=x, outputs=y)
    return unet
