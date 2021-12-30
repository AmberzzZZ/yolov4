import keras.backend as K
import tensorflow as tf


def compute_loss(args, nC, nA, anchors, strides=[8,16,32]):

    nL = len(args) // 2
    preds = args[:nL]
    targets = args[nL:]

    lcls, lbox, lobj = [], [], []
    for i in range(nL):
        pred_cur = preds[i]
        target_cur = targets[i]

        # grid
        b, grid_h, grid_w = pred_cur.shape[:3]
        grid_coord_x, grid_coord_y = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))  # [h,w]
        grid_coord_yx = tf.cast(tf.reshape(tf.stack([grid_coord_y, grid_coord_x]), (1,grid_h,grid_w,1,2)), tf.float32)   # [1,h,w,1,2]

        # reshape
        pred_cur = tf.reshape(pred_cur, (-1,grid_h,grid_w,nA,nC+1+4))

        # decode
        pred_probs = K.sigmoid(pred_cur[...,:nC+1])  # [b,h,w,a,c+1]
        pred_xcyc = (K.sigmoid(pred_cur[...,-4:-2])*2-0.5 + grid_coord_yx) * strides[i]
        pred_wh = (K.sigmoid(pred_cur[...,-2:])*2)**2 * anchors[i]
        pred_boxes = tf.concat([pred_xcyc,pred_wh], axis=-1)

        # iou: on positives
        iou = compute_iou(pred_boxes, target_cur[...,nC:])   # [b,h,w,a,1]
        iou_loss = 1 - iou   # [-1,1]->[0,2]
        lbox.append(K.sum(iou_loss, axis=[1,2,3,4]) / (K.sum(target_cur[...,nC:nC+1], axis=[1,2,3,4])+1))

        # regress loss: mean GIOU on positives
        lbox.append(K.sum(iou, axis=[1,2,3,4]) / (K.sum(target_cur[...,nC])+1))

        # cls loss: BCE on positives
        lcls.append(bce_loss(pred_probs[...,:nC], target_cur[...,:nC]))

        # obj loss: BCE on all
        lobj.append(bce_loss(pred_probs[...,:nC:nC+1], tf.stop_gradient(iou)))

    lcls = tf.add_n(lcls)   # [b,]
    lbox = tf.add_n(lbox)
    lobj = tf.add_n(lobj)

    loss = lcls + lbox + lobj
    loss = tf.Print(loss, [lcls,lbox,lobj], message='loss cls, box, obj')

    return loss


def bce_loss(pred, gt):

    pt = 1 - K.abs(gt-pred)
    pt = K.clip(pt, K.epsilon(), 1-K.epsilon())
    loss = -K.log(pt)
    return K.mean(loss, axis=[1,2,3,4])


def compute_iou(pred, gt):

    obj = gt[...,0]   # [b,h,w,a]
    positives = tf.where(obj>0)

    pred_boxes = tf.gather_nd(pred, positives)
    gt_boxes = tf.gather_nd(gt[...,1:], positives)

    # xcycwh -> x1y1x2y2
    pred_x1y1 = pred_boxes[:,:2] - pred_boxes[:,2:]/2.
    pred_x2y2 = pred_boxes[:,:2] + pred_boxes[:,2:]/2.
    pred_x1, pred_y1 = tf.split(pred_x1y1, 2, axis=-1)
    pred_x2, pred_y2 = tf.split(pred_x2y2, 2, axis=-1)

    gt_x1y1 = gt_boxes[:,:2] - gt_boxes[:,2:]/2.
    gt_x2y2 = gt_boxes[:,:2] + gt_boxes[:,2:]/2.
    gt_x1, gt_y1 = tf.split(gt_x1y1, 2, axis=-1)
    gt_x2, gt_y2 = tf.split(gt_x2y2, 2, axis=-1)

    # iou
    inter_w = K.maximum(0., K.minimum(gt_x2, pred_x2) - K.maximum(gt_x1, pred_x1))
    inter_h = K.maximum(0., K.minimum(gt_y2, pred_y2) - K.maximum(gt_y1, pred_y1))
    inter = inter_w * inter_h

    area_gt = K.maximum(0., (gt_x2-gt_x1)*(gt_y2-gt_y1))
    area_pred = K.maximum(0., (pred_x2-pred_x1)*(pred_y2-pred_y1))
    union = area_gt + area_pred - inter + 1e-16

    iou = inter / union

    # giou
    convex_w = K.maximum(gt_x2, pred_x2) - K.minimum(gt_x1, pred_x1)
    convex_h = K.maximum(gt_y2, pred_y2) - K.minimum(gt_y1, pred_y1)
    c_area = convex_w * convex_h + 1e-16

    giou = iou - (c_area - union) / c_area

    # structural
    structural_iou = tf.scatter_nd(positives, giou, tf.cast(tf.shape(gt[...,0:1]), tf.int64))

    return structural_iou


if __name__ == '__main__':
    
    from keras.layers import Lambda, Input, Dense
    from keras.models import Model
    from config import get_config
    import numpy as np

    cfg = get_config(None)
    h = w = 640
    nA = 3
    nC = 80
    strides = [8,16,32]

    inputs = [Input((h//s,w//s,nA*(nC+1+4))) for s in strides]
    preds = [Dense(nA*(nC+1+4))(i) for i in inputs]
    targets = [Input((h//s,w//s,nA,nC+1+4)) for s in strides]
    loss = Lambda(compute_loss, arguments={'nC':nC,'nA':nA,'anchors':cfg.ANCHOR.ANCHORS})([*preds,*targets])
    model = Model(inputs+targets, loss)
    model.compile('sgd', loss=lambda y_true,y_pred: y_pred)

    X = [np.ones((2,h//s,w//s,nA*(nC+1+4))) for s in strides]
    Y = [np.ones((2,h//s,w//s,nA,nC+1+4)) for s in strides]
    Z = np.ones((2,))
    model.fit([*X,*Y], Z)




