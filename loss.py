import keras.backend as K
import tensorflow as tf
import numpy as np


def yolo_loss(args, anchors, n_classes, input_shape, train_stage=1, iou_ratio=1.0):
    # gt & pred: [b,h,w,a,4+c+1]
    n_levels = len(args) // 2
    n_anchors = anchors.shape[0] // n_levels
    preds = args[:n_levels]
    gts = args[n_levels:]

    balance = [4.0, 1.0, 0.4, 0.1] if n_levels<5 else [4.0, 1.0, 0.5, 0.4, 0.1]
    loss = 0.
    for i in range(n_levels):
        pred = preds[i]
        gt = gts[i]

        pos_mask = tf.where(gt[...,-1:]>0, tf.ones_like(gt[...,-1:]), tf.zeros_like(gt[...,-1:]))     # [b,h,w,a]
        # cls loss: bce
        cls_pred = pred[...,4:4+n_classes]
        cls_gt = gt[...,4:4+n_classes]
        cls_loss_ = cls_loss(cls_gt, cls_pred) * pos_mask
        cls_loss_ = K.sum(cls_loss_, axis=[1,2,3,4])
        cls_loss_ = K.mean(cls_loss_)

        h,w,a = K.int_shape(pos_mask)[1:4]
        grid_xy = np.meshgrid(np.arrange(h), np.arrange(w))
        grid_xy = np.stack(grid_xy, axis=-1).reshape((1,h,w,1,2))
        grid_xy = tf.constant(grid_xy, dtype='float32')
        box_pred = pred[...,:4]
        box_gt = gt[...,:4]
        box_xywh = offset2abs(box_pred, anchors[i*n_anchors:(i+1)*n_anchors], grid_xy, strides[i], input_shape)

        if train_stage==1:
            # initial stage: hard conf + straight box regression, filter most of bg
            # reg loss: mse
            box_loss_ = K.square(box_gt - box_xywh) * pos_mask
            box_loss_ = K.sum(box_loss_, axis=[1,2,3,4])
            box_loss_ = K.mean(box_loss_)
            # conf loss: mse
            conf_loss_ = K.square(gt[...,-1:] - pred[...,-1:]) * pos_mask
            conf_loss_ = K.sum(conf_loss_, axis=[1,2,3,4])
            conf_loss_ = K.mean(conf_loss_)

        else:
            # refined stage: soft conf + refined box regression
            # reg loss: giou
            box_xywh = K.reshape(box_xywh, (-1,4))
            box_gt = K.reshape(gt[...,:4], (-1,4))
            iou = cal_iou(box_gt, box_xywh, GIoU=False, DIoU=False, CIoU=False)
            iou = K.reshape(iou, K.shape(pos_mask))
            box_loss_ = K.sum((1-iou)*pos_mask, axis=[1,2,3,4]) / K.sum(pos_mask)
            # conf loss: bce on giou
            conf_gt = iou
            conf_pred = pred[...,-1:]
            conf_loss_ = conf_loss_(conf_gt, conf_pred)
            conf_loss_ = K.sum(conf_loss_, axis=[1,2,3,4])
            conf_loss_ = K.mean(conf_loss_)

        loss += (cls_loss_ + box_loss_ + conf_loss_) * balance[i]

    return loss


def cal_iou(box_true, box_pred, GIoU=False, DIoU=False, CIoU=False):
    # box_gt & pred: [N,4], row-matching rela-origin-abs-normed-xcycwh
    gt_x1y1 = box_true[:,:2] - box_true[:,2:]/2.
    gt_x1, gt_y1 = tf.split(gt_x1y1, 2, axis=-1)
    gt_x2y2 = box_true[:,:2] + box_true[:,2:]/2.
    gt_x2, gt_y2 = tf.split(gt_x2y2, 2, axis=-1)
    pred_x1y1 = box_pred[:,:2] - box_pred[:,2:]/2.
    pred_x1, pred_y1 = tf.split(pred_x1y1, 2, axis=-1)
    pred_x2y2 = box_pred[:,:2] + box_pred[:,2:]/2.
    pred_x2, pred_y2 = tf.split(pred_x2y2, 2, axis=-1)

    inter_w = K.maximum(0., K.minimum(gt_x2, pred_x2) - K.maximum(gt_x1, pred_x1))
    inter_h = K.maximum(0., K.minimum(gt_y2, pred_y2) - K.maximum(gt_y1, pred_y1))
    inter = inter_w * inter_h

    area_gt = K.maximum(0., (gt_x2-gt_x1)*(gt_y2-gt_y1))
    area_pred = K.maximum(0., (pred_x2-pred_x1)*(pred_y2-pred_y1))
    union = area_gt + area_pred - inter + 1e-16

    iou = inter / union

    return iou


def offset2abs(txywh, anchors, grid_xy, stride, input_shape):
    # txywh: [b,h,w,a,4]
    # anchors: [a,2]
    # return: rela-origin-normed-xcycwh
    anchors = tf.constant(anchors, dtype='float32')
    txy = txywh[...,:2]
    pxy = (K.sigmoid(txy)*2 - 0.5 + grid_xy) * stride / input_shape
    twh = txywh[...,2:]
    pwh = K.pow(K.sigmoid(twh)*2, 2) * K.reshape(anchors, (1,1,1,-1,2)) / input_shape
    return K.concatenate([pxy,pwh], axis=-1)


def cls_loss(y_true, y_pred):
    # bce from logits
    y_pred = K.sigmoid(y_pred)
    pt = 1 - K.abs(y_true-y_pred)
    pt = K.clip(pt, K.epsilon(), 1-K.epsilon())
    loss = -K.log(pt)
    return loss


def conf_loss(y_true, y_pred, ignore_thresh=0.3):
    # bce from logits
    y_pred = K.sigmoid(y_pred)
    pt = 1 - K.abs(y_true-y_pred)
    pt = K.clip(pt, K.epsilon(), 1-K.epsilon())
    loss = -K.log(pt)
    loss = tf.where(y_true>ignore_thresh, loss, tf.zeros_like(loss))
    return loss







