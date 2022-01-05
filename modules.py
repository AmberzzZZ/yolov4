from keras.layers import Conv2D, BatchNormalization, Activation, multiply, concatenate, MaxPooling2D
from keras.models import Model
import keras.backend as K
import math


def mish(x):
    # f(x) = x * tanh(softplus(x))
    # softplus(x) = ln(1+exp(x))
    return multiply([x, K.tanh(K.log(1+K.exp(x)))])


class Conv(Model):

    def __init__(self, n_filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                 act=True, **kargs):
        super(Conv, self).__init__(**kargs)

        self.act = act
        self.strides = strides
        self.n_filters = n_filters

        self.conv = Conv2D(n_filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)
        self.bn = BatchNormalization(axis=-1, momentum=0.97, epsilon=1e-3)
        if self.act:
            self.mish = Activation(mish)

    def call(self, x, training=None):

        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.act:
            x = self.mish(x)

        return x

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        new_h = math.ceil(h/self.strides)
        new_w = math.ceil(w/self.strides)
        return (b, new_h, new_w, self.n_filters)


class Bottleneck(Model):

    # darknet bottleneck: 1x1conv(c//2) - 3x3conv(c) with id path

    def __init__(self, n_filters, shortcut=True, expand_ratio=0.5, **kargs):
        super(Bottleneck, self).__init__(**kargs)

        self.n_filters = n_filters
        self.shortcut = shortcut

        mid_filters = int(n_filters*expand_ratio)

        self.conv1 = Conv(mid_filters, 1, strides=1)
        self.conv2 = Conv(n_filters, 3, strides=1)

    def call(self, x, training=None):

        inpt = x
        x = self.conv2(self.conv1(x,training=training),training=training)
        if self.shortcut and self.n_filters==x.shape[-1]:
            return x + inpt
        else:
            return x

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (b, h, w, self.n_filters)


class BottleneckCSP(Model):

    def __init__(self, n_filters, n_blocks, shortcut=True, expand_ratio=0.5, **kargs):
        super(BottleneckCSP, self).__init__(**kargs)

        self.n_filters = n_filters
        self.n_blocks = n_blocks

        mid_filters = int(n_filters*expand_ratio)

        self.conv1 = Conv(mid_filters, 1, strides=1)
        self.conv2 = Conv2D(mid_filters, 1, strides=1, padding='same', use_bias=False)
        self.conv3 = Conv2D(mid_filters, 1, strides=1, padding='same', use_bias=False)
        self.conv4 = Conv(n_filters, 1, strides=1)
        self.bn = BatchNormalization(axis=-1, momentum=0.97, epsilon=1e-3)
        self.mish = Activation(mish)
        self.bottlenecks = []
        for i in range(n_blocks):
            self.tmp = Bottleneck(mid_filters, shortcut, expand_ratio=1.0)
            self.bottlenecks.append(self.tmp)

    def call(self, x, training=None):

        # residual path
        y1 = self.conv1(x, training=training)
        for b in self.bottlenecks:
            y1 = b(y1)
        y1 = self.conv3(y1, training=training)

        # id path
        y2 = self.conv2(x)

        # concat
        y = concatenate([y1,y2])
        y = self.bn(y, training=training)
        y = self.mish(y)
        y = self.conv4(y, training=training)

        return y

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (b, h, w, self.n_filters)


class SPPCSP(Model):

    def __init__(self, n_filters, n_blocks=1, expand_ratio=0.5, kernels=(5, 9, 13), **kargs):
        super(SPPCSP, self).__init__(**kargs)

        self.n_filters = n_filters

        mid_filters = int(n_filters*expand_ratio*2)

        self.conv1 = Conv(mid_filters, 1, strides=1)
        self.conv2 = Conv2D(mid_filters, 1, strides=1, use_bias=False)
        self.conv3 = Conv(mid_filters, 3, strides=1)
        self.conv4 = Conv(mid_filters, 1, strides=1)
        self.poolings = []
        for k in kernels:
            self.tmp = MaxPooling2D(pool_size=k, strides=1, padding='same')
            self.poolings.append(self.tmp)
        self.conv5 = Conv(mid_filters, 1, strides=1)
        self.conv6 = Conv(mid_filters, 3, strides=1)
        self.bn = BatchNormalization(axis=-1, momentum=0.97, epsilon=1e-3)
        self.mish = Activation(mish)
        self.conv7 = Conv(n_filters, 1, strides=1)

    def call(self, x, training=training):

        # residual path
        x1 = self.conv4(self.conv3(self.conv1(x,training=training),training=training),training=training)
        x2,x3,x4 = [p(x1) for p in self.poolings]
        y1 = concatenate([x1,x2,x3,x4])
        y1 = self.conv6(self.conv5(y1))

        # id path
        y2 = self.conv2(x)

        # concat
        y = concatenate([y1,y2])
        y = self.bn(y, training=training)
        y = self.mish(y)
        y = self.conv7(y, training=training)

        return y

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (b, h, w, self.n_filters)


class BottleneckCSP2(Model):

    def __init__(self, n_filters, n_blocks, shortcut=False, expand_ratio=0.5, **kargs):
        super(BottleneckCSP2, self).__init__(**kargs)

        self.n_filters = n_filters
        self.n_blocks = n_blocks

        self.conv1 = Conv(n_filters, 1, strides=1)
        self.conv2 = Conv2D(n_filters, 1, strides=1, padding='same', use_bias=False)
        self.conv3 = Conv(n_filters, 1, strides=1)
        self.bn = BatchNormalization(axis=-1, momentum=0.97, epsilon=1e-3)
        self.mish = Activation(mish)
        self.bottlenecks = []
        for i in range(n_blocks):
            self.tmp = Bottleneck(n_filters, shortcut, expand_ratio=1.0)
            self.bottlenecks.append(self.tmp)

    def call(self, x, training=None):

        x = self.conv1(x)

        # residual path
        y1 = x
        for b in self.bottlenecks:
            y1 = b(y1, training=training)

        # id path
        y2 = self.conv2(x)

        # concat
        y = concatenate([y1,y2])
        y = self.bn(y, training=training)
        y = self.mish(y)
        y = self.conv3(y, training=training)

        return y

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (b, h, w, self.n_filters)


