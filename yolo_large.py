from keras.layers import Input, UpSampling2D, concatenate, Conv2D, Lambda, Reshape
from keras.models import Model
from utils import mish, conv_BN, bottleneckCSP, bottleneckCSP1, CSPSPP
from loss import yolo_loss


n_blocks = [1,3,15,15,7,7,7]
n_filters = [64, 128, 256, 512, 1024, 1024, 1024]
n_stages = {'P5': 5, 'P6': 6, 'P7': 7}


def yolov4(anchors, input_shape=(896,896,3), n_classes=80, scale='P5'):

    inpt = Input(input_shape)

    # stem
    x = conv_BN(inpt, 32, 3, strides=1, activation=mish)
    x = conv_BN(x, 64, 3, strides=2, activation=mish)

    # back stages: x2 - x128
    features = []
    for s in range(n_stages[scale]):
        n_resblocks = n_blocks[s]
        n_cspfilters = n_filters[s]
        if s>0:
            # downsamp
            x = conv_BN(x, n_cspfilters, 3, strides=2, activation=mish)
        # cspblock
        x = bottleneckCSP(x, n_cspfilters, n_resblocks)

        if s>1:
            if s!= n_stages[scale]-1:
                # conv-bn-act to transfer task
                x = conv_BN(x, n_cspfilters//2, 1, strides=1, activation=mish)
            else:
                # use cspspp
                x = CSPSPP(x, n_cspfilters//2, pool_size=(5,9,13))
            features.append(x)

    # pan: bottom-up & top-down
    features = pan(features, n_stages[scale])

    # head: shared
    n_levels = len(features)
    n_anchors = anchors.shape[0] // n_levels
    outputs = [Conv2D(n_anchors*(n_classes+5), 1, strides=1, padding='same')(i) for i in features]

    # loss
    h, w, _ = input_shape
    strides = {0:8, 1:16, 2:32, 3:64, 4:128}
    gt = [Input((h//strides[i], w//strides[i], n_anchors, n_classes+5)) for i in range(n_levels)]
    outputs = [Reshape(((h//strides[i], w//strides[i], n_anchors, n_classes+5)))(outputs[i]) for i in range(n_levels)]
    loss = Lambda(yolo_loss, arguments={'anchors': anchors, 'n_classes': n_classes})(outputs+gt)

    model = Model([inpt, *gt], loss)

    return model


def pan(features, n_stages):
    n_levels = len(features)

    fpn_features = []     # min to max
    # bottom-up
    for l in range(n_levels-1, -1, -1):
        conv = features[l]
        n_fpnfilters = n_filters[l+2]//2
        if l==n_levels-1:
            fpn_features.append(conv)
        else:
            up = fpn_features[-1]
            up = conv_BN(up, n_fpnfilters, 1, strides=1, activation=mish)
            up = UpSampling2D(size=2)(up)
            x = concatenate([conv, up])
            x = bottleneckCSP1(x, n_fpnfilters, 3)
            fpn_features.append(x)

    # print("fpn: ", fpn_features)

    pan_features = []    # max to min
    # top-down
    for l in range(n_levels-1, -1, -1):
        pconv = fpn_features[l]
        n_panfilters = n_filters[n_stages-l-1]
        if l==n_levels-1:
            x = conv_BN(x, n_panfilters, 3, strides=1, activation=mish)
            pan_features.append(x)
        else:
            down = pan_features[-1]
            down = conv_BN(down, n_panfilters, 3, strides=2, activation=mish)
            x = concatenate([pconv, down])
            x = bottleneckCSP1(x, n_panfilters, 3)
            x = conv_BN(x, n_panfilters, 3, strides=1, activation=mish)
            pan_features.append(x)

    # print("pan: ", pan_features)

    return pan_features


if __name__ == '__main__':

    import numpy as np
    anchors = np.zeros((9,2))
    model = yolov4(anchors, scale='P5')
    model.summary()








