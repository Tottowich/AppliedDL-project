import tensorflow as tf
from segmentation.encoder import build_encoder#, initialize_encoder

from segmentation.decoders import build_decoder#, initialize_decoder
from segmentation.unet import build_unet
from tensorflow.keras.layers import Input
# Plot model
from tensorflow.keras.utils import plot_model
# H = 256
# W = 256
# C = 1
# x = Input(shape=(H,W,C))
# #x = tf.random.normal((1,H,W,C))
# filters = [16,32,64,128,256,512]
# encoder = build_encoder((H,W,C),filters=filters, kernel_size=3, strides=1, padding="same", activation="relu", depth=[3,2,1], drop_rate=0.0)
# xs = encoder(x)
# print(encoder.summary(expand_nested=True))
# # plot_model(model=encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True,expand_nested=True)
# decoder = build_decoder(xs, num_classes=4,filters=filters[::-1][1:], kernel_size=3, strides=2, padding="same", activation="relu", depth=0, drop_rate=0.0)
# print(decoder.summary(expand_nested=True))
# # plot_model(model=decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True,expand_nested=True)
# # Generate decoder inputs from encoder outputs
# y = decoder(xs[::-1])
# # unet = tf.keras.Model(inputs=x, outputs=y)
# # print(unet.summary(expand_nested=True))
# # plot_model(model=unet, to_file='unet.png', show_shapes=True, show_layer_names=True,expand_nested=True)
# unet = tf.keras.Model(inputs=x, outputs=y)
# print(unet.summary(expand_nested=True))
# plot_model(model=unet, to_file='unet.png', show_shapes=True, show_layer_names=True,expand_nested=False)

H = 256
W = 256
C = 1
input_shape = (H,W,C)
num_classes = 2
filters = [8,16,32,64,128,256]
kernel_size = 3
strides = 1
padding = "same"
activation = "relu"
drop_rate_encoder = [0.0]
drop_rate_decoder = [0.0]
depth_encoder = [1]
depth_decoder = [0,1]
output_depth = 1
output_activation = "sigmoid"

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
)
print(unet.summary(expand_nested=True))
plot_model(model=unet, to_file='unet.png', show_shapes=True, show_layer_names=True,expand_nested=True)
# Test with random input
x = tf.random.normal((1,H,W,C))
y = unet(x)
print(y.shape)
# Plot the restults side by side
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(x[0,:,:,0])
plt.subplot(1,2,2)
plt.imshow(y[0,:,:,0])
plt.show()
