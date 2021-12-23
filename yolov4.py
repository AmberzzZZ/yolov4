from modules import *
from keras.layers import Input, UpSampling2D, concatenate


def yolov4(input_shape=(896,896,3), n_classes=80, n_anchors=3):

    inpt = Input(input_shape)

    # stem
    x = Conv(32, 3, strides=1)(inpt)
    x = Conv(64, 3, strides=2)(x)        # x2

    # 5 stages: x4-x32
    feats = []
    n_filters = 64
    n_blocks = [1,3,15,15,7]
    for i in range(5):
        # bottleneck
        x = BottleneckCSP(n_filters*(2**i), n_blocks[i])(x)
        feats.append(x)        # P1-P5, x2-x32, channel 64-1024
        # downsamp
        if i<4:
            x = Conv(n_filters*(2**(i+1)), 3, strides=2)(x)

    # fpn
    fpn_feats = fpn(feats[2:])    # (b,h,w,c)

    # head
    n_levels = len(fpn_feats)
    outputs = []
    for i in range(n_levels):
        x = Conv2D(n_anchors*(n_classes+1+4), 1, strides=1, activation='sigmoid')(fpn_feats[i])
        outputs.append(x)

    model = Model(inpt, outputs)

    return model


def fpn(feats):

    n_filters = int(feats[0].shape[-1])
    n_levels = len(feats)

    # up path
    up_feats = []
    for i in range(n_levels):
        x = feats[n_levels-1-i]
        base_filters = n_filters*2**(n_levels-i-1)

        if i==0:
            # SPPCSP
            x = SPPCSP(base_filters//2)(x)
        else:
            # upsamp
            up = up_feats[-1]
            up = Conv(base_filters//2, 1, strides=1)(up)
            up = UpSampling2D(size=2, interpolation='nearest')(up)
            # conv, concat, CSP
            x = Conv(base_filters//2, 1, strides=1)(x)
            x = concatenate([x, up])
            x = BottleneckCSP2(base_filters//2, 3)(x)

        up_feats.append(x)

    # down path
    down_feats = [up_feats[-1]]
    out_feats = []
    for i in range(n_levels):
        x = feats[n_levels-1-i]
        base_filters = n_filters*2**(i)

        x = down_feats[-1]
        # out conv
        out = Conv(base_filters, 3, strides=1)(x)
        out_feats.append(out)

        if i<n_levels-1:
            # down conv, concat, CSP
            down = Conv(base_filters, 3, strides=2)(x)
            down = concatenate([down, up_feats[n_levels-2-i]])
            down = BottleneckCSP2(base_filters, 3)(down)
            down_feats.append(down)

    return out_feats


if __name__ == '__main__':

    model = yolov4(input_shape=(896,896,3), n_classes=80)
    model.summary()







