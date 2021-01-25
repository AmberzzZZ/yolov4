from keras.layers import Conv2D, BatchNormalization, multiply, add, concatenate, MaxPooling2D
import keras.backend as K
from keras.engine.topology import Layer


class mish(Layer):

    def __init__(self, **kwargs):
            super(mish, self).__init__(**kwargs)

    def call(self, x):
        # f(x) = x * tanh(softplus(x))
        # softplus(x) = ln(1+exp(x))
        return multiply([x, K.tanh(K.log(1+K.exp(x)))])

    def compute_output_shape(self, input_shape):
        return input_shape


def conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = mish()(x)
    return x


def bottleneck(inpt, n_filters, skip=True):
    # 1x1 ch//2, 3x3 ch, id skip
    x = conv_BN(inpt, n_filters//2, 1, strides=1, activation=mish)
    x = conv_BN(x, n_filters, 3, strides=1, activation=mish)
    if skip and K.int_shape(inpt)[-1]==n_filters:
        return add([x, inpt])
    else:
        return x


def bottleneckCSP(inpt, n_filters, n_blocks):
    # CSP block in backbone
    # split
    x = conv_BN(inpt, n_filters//2, 1, strides=1, activation=mish)
    skip = Conv2D(n_filters//2, 1, strides=1, padding='same', activation=None, use_bias=False)(inpt)
    # res blocks
    for i in range(n_blocks):
        x = bottleneck(x, n_filters//2, skip=True)
    # inside transition: 1x1 conv
    x = Conv2D(n_filters//2, 1, strides=1, padding='same', activation=None, use_bias=False)(x)
    # concat
    x = concatenate([x, skip], axis=-1)
    x = BatchNormalization()(x)
    x = mish()(x)
    # outside transition: 1x1 conv-bn-act
    x = conv_BN(x, n_filters, 1, strides=1, activation=mish)
    return x


def bottleneckCSP1(inpt, n_filters, n_blocks):
    # CSP block in FPN: shared 1x1 bottle
    # split
    x = conv_BN(inpt, n_filters//2, 1, strides=1, activation=mish)
    skip = Conv2D(n_filters//2, 1, strides=1, padding='same', activation=None, use_bias=False)(x)
    # res blocks
    for i in range(n_blocks):
        x = bottleneck(x, n_filters//2, skip=True)
    # concat
    x = concatenate([x, skip], axis=-1)
    x = BatchNormalization()(x)
    x = mish()(x)
    # outside transition: 1x1 conv-bn-act
    x = conv_BN(x, n_filters, 1, strides=1, activation=mish)
    return x


def spp(inpt, n_filters, pool_size=(5,9,13)):
    # 1x1 conv, narrow
    x = conv_BN(inpt, n_filters//2, 1, strides=1, activation=mish)
    # spp
    pooled_features = [MaxPooling2D(pool_size=size, strides=1, padding='same')(x) for size in pool_size]
    # concat
    x = concatenate(pooled_features, axis=-1)
    # 1x1 conv, widen
    x = conv_BN(inpt, n_filters, 1, strides=1, activation=mish)
    return x


def CSPSPP(inpt, n_filters, pool_size=(5,9,13)):
    # split
    x = conv_BN(inpt, n_filters, 1, strides=1, activation=mish)
    skip = Conv2D(n_filters, 1, strides=1, padding='same', activation=None, use_bias=False)(inpt)
    # CSP block: 3x3 conv, 1x1 conv, spp, 3x3 conv
    x = conv_BN(x, n_filters, 3, strides=1, activation=mish)
    x = conv_BN(x, n_filters, 1, strides=1, activation=mish)
    pooled_features = [MaxPooling2D(pool_size=size, strides=1, padding='same')(x) for size in pool_size]
    x = concatenate([x]+pooled_features, axis=-1)
    x = conv_BN(x, n_filters, 1, strides=1, activation=mish)
    x = conv_BN(x, n_filters, 3, strides=1, activation=False)
    # concat
    x = concatenate([x, skip])
    x = BatchNormalization()(x)
    x = mish()(x)
    x = conv_BN(inpt, n_filters, 1, strides=1, activation=mish)
    return x


def focus(x, n_filters):
    # s2 resample
    x1 = x[:,::2,::2,:]
    x2 = x[:,1::2,::2,:]
    x3 = x[:,::2,1::2,:]
    x4 = x[:,1::2,1::2,:]
    # concat
    x = concatenate([x1,x2,x3,x4], axis=-1)
    x = conv_BN(x, n_filters, 1, strides=1, activation=mish)
    return x

























