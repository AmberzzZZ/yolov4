from modules import *
from keras.layers import Input, UpSampling2D, concatenate, Lambda, Activation
from loss import compute_loss


def yolov4(input_shape=(896,896,3), n_classes=80, n_anchors=3, cfg=None, training=1, test=False):

    inpt = Input(input_shape)

    # stem
    x = Conv(32, 3, strides=1)(inpt, training=training)
    x = Conv(64, 3, strides=2)(x, training=training)        # x2

    # 5 stages: x4-x32
    feats = []
    n_filters = 64
    n_blocks = [1,3,15,15,7]
    for i in range(5):
        # bottleneck
        x = BottleneckCSP(n_filters*(2**i), n_blocks[i])(x, training=training)
        feats.append(x)        # P1-P5, x2-x32, channel 64-1024
        # downsamp
        if i<4:
            x = Conv(n_filters*(2**(i+1)), 3, strides=2)(x, training=training)

    # fpn
    fpn_feats = fpn(feats[2:], training=training)    # (b,h,w,c)

    # head
    n_levels = len(fpn_feats)
    outputs = []
    for i in range(n_levels):
        x = Conv2D(n_anchors*(n_classes+1+4), 1, strides=1, activation=None)(fpn_feats[i])
        outputs.append(x)

    if test:
        outputs = [Activation('sigmoid')(o) for o in outputs]
        model = Model(inpt, outputs)
    else:
        # loss
        loss = Lambda(compute_loss, arguments={'nC':n_classes, 'nA':n_anchors, 'anchors':cfg.ANCHOR.ANCHORS},
                      name='yolo_loss')([*outputs, *targets])
        model = Model(inpt, loss)

    return model


def fpn(feats, training=None):

    n_filters = int(feats[0].shape[-1])
    n_levels = len(feats)

    # up path
    up_feats = []
    for i in range(n_levels):
        x = feats[n_levels-1-i]
        base_filters = n_filters*2**(n_levels-i-1)

        if i==0:
            # SPPCSP
            x = SPPCSP(base_filters//2)(x, training=training)
        else:
            # upsamp
            up = up_feats[-1]
            up = Conv(base_filters//2, 1, strides=1)(up, training=training)
            up = UpSampling2D(size=2, interpolation='nearest')(up)
            # conv, concat, CSP
            x = Conv(base_filters//2, 1, strides=1)(x, training=training)
            x = concatenate([x, up])
            x = BottleneckCSP2(base_filters//2, 3)(x, training=training)

        up_feats.append(x)

    # down path
    down_feats = [up_feats[-1]]
    out_feats = []
    for i in range(n_levels):
        base_filters = n_filters*2**(i)
        x = down_feats[-1]

        # out conv
        out = Conv(base_filters, 3, strides=1)(x, training=training)
        out_feats.append(out)

        if i<n_levels-1:
            # down conv, concat, CSP
            down = Conv(base_filters, 3, strides=2)(x, training=training)
            down = concatenate([down, up_feats[n_levels-2-i]])
            down = BottleneckCSP2(base_filters, 3)(down, training=training)
            down_feats.append(down)

    return out_feats


if __name__ == '__main__':

    model = yolov4(input_shape=(896,896,3), n_classes=80)
    # model.load_weights("weights/yolov4-p5.h5", by_name=True)
    # model.summary()

    import numpy as np
    model.compile('sgd', loss='binary_crossentropy')
    X = np.ones((4,896,896,3))
    strides = [8,16,32]
    h = w = 896
    base = 256
    Y = [np.ones((4, h//s, w//s, base*(2**i))) for i,s in enumerate(strides)]
    model.fit(X, Y)






