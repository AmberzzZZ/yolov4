from PIL import Image, ImageFont, ImageDraw
import cv2
import os
import numpy as np
from yolov4 import yolov4
from config import get_config
from nms import hard_nms, giou_nms
from utils import get_classes


if __name__ == '__main__':

    test_dir = "data/test/"

    # args
    cfg = get_config(None)
    img_size = cfg.DATA.IMG_SIZE
    n_classes = cfg.MODEL.NUM_CLASSES
    n_anchors = cfg.ANCHOR.NA
    anchors = cfg.ANCHOR.ANCHORS
    strides = cfg.ANCHOR.STRIDES
    classes = get_classes('classes.txt')

    # model
    model = yolov4(input_shape=(img_size,img_size,3), n_classes=n_classes, n_anchors=n_anchors,
                   cfg=cfg, training=1, test=True)
    model.load_weights('weights/yolov4-p5.h5')

    # inference
    for file in os.listdir(test_dir):

        cv_img = cv2.imread(os.path.join(test_dir, file), 1)
        h0,w0 = cv_img.shape[:2]
        r = img_size / max(h0,w0)
        if r!=1:
            cv_img = cv2.resize(cv_img, (int(w0*r), int(h0*r)), interpolation=cv2.INTER_LINEAR)
        h,w = cv_img.shape[:2]
        pad_h, pad_w = img_size - h, img_size - w
        cv_img = np.pad(cv_img, [[0,pad_h],[0,pad_w],[0,0]], mode='constant', constant_values=114)

        inpt = np.expand_dims(cv_img[:,:,::-1], axis=0)/255.
        predictions = model.predict(inpt)    # list of [b,h,w,a,c+1+4]

        nL = len(predictions)
        boxes_xcyc_postives_all_level = []  # abs
        boxes_wh_postives_all_level = []   # abs
        conf_postives_all_level = []
        probs_postives_all_level = []
        for i in range(nL):
            grid_h = grid_w = img_size//strides[i]
            pred = np.reshape(predictions[i], (grid_h,grid_w,n_anchors,n_classes+1+4))
            probs = pred[...,:n_classes]
            conf = pred[...,n_classes]    # [h,w,a]
            boxes_xcyc = pred[...,-4:-2] * 2 - 0.5
            boxes_wh = (pred[...,-2:]*2)**2

            pred_positives = np.where(conf>0.1)   # [y-indices,x-indices,a-indices]

            probs_positives = probs[pred_positives]
            conf_postives = conf[pred_positives]
            boxes_xcyc_postives = boxes_xcyc[pred_positives]
            boxes_wh_postives = boxes_wh[pred_positives]
            anchor_positives = anchors[i][pred_positives[2]]

            grid_cx, grid_cy = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            grid_xy = np.stack([grid_cx,grid_cy], axis=2).reshape((grid_h,grid_w,1,2))
            grid_xy = np.tile(grid_xy, [1,1,n_anchors,1])
            grid_positives = grid_xy[pred_positives]   # [N,2]

            # offset 2 abs
            boxes_xcyc_postives = (boxes_xcyc_postives+grid_positives) * strides[i]
            boxes_wh_postives = boxes_wh_postives * anchor_positives

            boxes_xcyc_postives_all_level.append(boxes_xcyc_postives)
            boxes_wh_postives_all_level.append(boxes_wh_postives)
            conf_postives_all_level.append(conf_postives)
            probs_postives_all_level.append(probs_positives)

        # merge all level
        boxes_xcyc_postives = np.concatenate(boxes_xcyc_postives_all_level, axis=0)
        boxes_wh_postives = np.concatenate(boxes_wh_postives_all_level, axis=0)
        conf_postives = np.concatenate(conf_postives_all_level, axis=0)
        probs_positives = np.concatenate(probs_postives_all_level, axis=0)

        # run nms across level
        boxes_x1y1 = boxes_xcyc_postives - boxes_wh_postives/2.
        boxes_x2y2 = boxes_xcyc_postives + boxes_wh_postives/2.
        boxes = np.concatenate([boxes_x1y1,boxes_x2y2],axis=1)
        scores = conf_postives * np.max(probs_positives, axis=1)
        labels = np.argmax(probs_positives, axis=1)
        if len(boxes):
            boxes, scores, labels = hard_nms(boxes,scores,labels,score_thresh=0.1,iou_thresh=0.3)

            # iter boxes
            f = open(os.path.join('preds/', file.replace('.jpg','.txt')), 'w')
            for b in range(len(boxes)):
                x1,y1,x2,y2 = boxes[b]
                prob = scores[b]
                label = labels[b]

                normed_xc = (x1+x2)/2./(img_size-pad_w)
                normed_yc = (y1+y2)/2./(img_size-pad_h)
                normed_w = (x2-x1)/(img_size-pad_w)
                normed_h = (y2-y1)/(img_size-pad_h)
                f.write(' '.join(map(str, [label,normed_xc,normed_yc,normed_w,normed_h])))

                # draw box
                cv2.rectangle(cv_img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,255), 3)
                pil_image = Image.fromarray(cv_img.astype(np.uint8))
                label = '{} {:.2f}'.format(classes[label], prob)
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype('msyh.ttc', size=np.floor(3e-2*pil_image.size[1]+0.5).astype('int32'))
                draw.text((int(x1),int(y1)), label, fill='blue', font=font)
                cv_img = np.array(pil_image)

            f.close()

        cv2.imwrite('preds/%s' % file, cv_img)













