import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
import random
import cv2
import math


# dataloader
def build_loader(config):

    # dataset
    dataset_train = CustomDataset(config, augment=True)

    # sampler
    sampler_train = SubsetRandomSampler(np.arange(0,len(dataset_train)))

    if config.DATA.BATCH_SIZE>len(sampler_train):
        config.defrost()
        config.DATA.BATCH_SIZE = len(sampler_train)
        config.freeze()

    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        collate_fn=CustomDataset.collate_fn,
    )

    return data_loader_train


# dataset
class CustomDataset(Dataset):

    def __init__(self, config, rect=False, augment=False, stride=32, pad=0.):

        # load all samples
        data_dir = config.DATA.DATA_PATH
        label_dir = config.DATA.LABEL_PATH
        batch_size = config.DATA.BATCH_SIZE
        self.img_size = config.DATA.IMG_SIZE
        self.img_channels = config.DATA.IMG_CHANNELS
        self.hyp = {'mixup':config.AUG.MIXUP,
                          'hsv':config.AUG.HSV,
                          'degrees':config.AUG.DEGREE, 'translate': config.AUG.TRANSLATE, 'scale':config.AUG.SCALE, 'shear':config.AUG.SHEAR,
                          'perspective':config.AUG.PERSPECTIVE,
                          'flipud':config.AUG.FLIPUD, 'fliplr':config.AUG.FLIPLR}
        self.augment = augment
        self.mosaic = augment and not rect
        self.rect = rect

        self.img_files = [i for i in glob.glob(data_dir + '/*jpg')]
        self.label_files = [i.replace(data_dir, label_dir).replace('jpg', 'txt') for i in self.img_files]

        self.n_samples = len(self.img_files)
        if os.path.isfile(config.DATA.DATASET+'.cache'):
            cache = torch.load(config.DATA.DATASET+'.cache')        # {img_pt: [boxes, img_shape]}
        else:
            cache = self.check_samples(config.DATA.DATASET+'.cache')

        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)     # [n_samples, 2], [w,h] for each sample
        self.labels = list(labels)   # list of boxes, [N,5] for each sample

        if self.rect:
            # sort & aggregate samples by h/w ratio
            s = self.shapes
            ratio = s[:,1] / s[:,0]
            indices = ratio.argsort()
            self.img_files = [self.img_files[i] for i in indices]
            self.label_files = [self.label_files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.shapes = s[indices]  # wh
            ratio = ratio[indices]

            # compute input_shape for each batch
            bi = np.floor(np.arange(self.n_samples) / batch_size).astype(np.int)  # batch index
            nb = bi[-1] + 1  # number of batches
            shapes = [[1, 1]] * nb
            for i in range(nb):
                batch_ratio = ratio[bi == i]
                mini, maxi = batch_ratio.min(), batch_ratio.max()
                if maxi < 1:   # h<w, h is the long side
                    shapes[i] = [maxi, 1]
                elif mini > 1:   # w>h, w is the long side
                    shapes[i] = [1, 1 / mini]

            # times of output strides
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

    def check_samples(self, cache_name=None):
        pbar = tqdm(zip(self.img_files, self.label_files))
        x = {}  # img: [boxes, img_shape]
        for (img, label) in pbar:
            try:
                boxes = []
                image = Image.open(img)
                image.verify()  # PIL verify
                shape = image.size  # image size, wh
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        # [N,5], [cls,xc,yc,w,h]
                        boxes = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                if len(boxes) == 0:
                    boxes = np.zeros((0, 5), dtype=np.float32)
                x[img] = [boxes, shape]
            except Exception as e:
                x[img] = None
                print('WARNING: %s: %s' % (img, e))

        torch.save(x, cache_name)   # for quickly reuse
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        # returns: img: tensor, (CHW), mutiples of 32
        #          boxes: tensor, (n_boxes,6)
        #          img_name: str, current file name
        #          shapes: tuple, only make sense in for single img

        hyp = self.hyp

        if self.mosaic:
            # load mosaic img & labels
            img, labels = load_mosaic(self, index)
            shapes = None

            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, self.n_samples))
                r = np.random.beta(8., 8.)
                img = (img * r + img2 * (1-r)).astype(np.uint8)
                labels = np.concatenate([labels, labels2], axis=0)

        else:
            # load single img
            img, (h0,w0), (h,w) = load_image(self, index)   # resized img, orig hw, resized hw
            img, ratio, pad = letterbox(img, (h,w), auto=False, scaleup=self.augment)  # pad to 32-multiples
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # orig hw, use ratio, wh padding for single sides

            # load boxes
            x = self.labels[index]   # [N,5] array, normed [clsid,xc,yc,w,h]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        # img: [0,255] cv_img, bgr
        # labels: abs [cls,x1,y1,x2,y2]

        # further augment
        if self.augment:

            # images space
            if not self.mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'],
                                                 )
            # color space
            img = augment_hsv(img, *hyp['hsv'])

        nL = len(labels)
        if nL:
            # convert to array [nL,5], normed [cls,xc,yc,w,h]
            labels[:,[3,4]] -= labels[:,[1,2]]   # abs_wh
            labels[:,[1,2]] += labels[:,[3,4]]/2.  # abs_xcyc
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        # flip
        if random.random() < hyp['flipud']:
            img = np.flipud(img)
            if nL:
                labels[:,2] = 1 - labels[:,2]

        if random.random() < hyp['fliplr']:
            img = np.fliplr(img)
            if nL:
                labels[:,1] = 1 - labels[:,1]

        # convert to tensor
        img = img[:,:,::-1].transpose((2,0,1))/255.   # BGR->RGB, HWC->CHW
        img = torch.from_numpy(img)

        labels_out = torch.zeros((nL,6))
        if nL:
            labels_out[:,1:] = torch.from_numpy(labels)

        return img, labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def load_mosaic(self, index):

    s = self.img_size
    # canvas
    canvas = np.full((s*2, s*2, self.img_channels), 114, dtype=np.uint8)
    canvas_boxes = []
    yc = xc = s    # mosaic center
    # load extra 3 images
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]
    for i, index in enumerate(indices):
        # Load image: resized img, with long side = s, resized_hw
        img, _, (h, w) = load_image(self, index)

        # place img in canvas
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # canvas range
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # img range
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        # Labels
        x = self.labels[index]   # boxes, (N,5) array, normed [cls_id, xc, yc, w, h]
        if x.size > 0:
            labels = x.copy()
            # img shift
            padw = x1a - x1b
            padh = y1a - y1b
            # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            canvas_boxes.append(labels)

    # Concat/clip labels
    if len(canvas_boxes):
        canvas_boxes = np.concatenate(canvas_boxes, 0)   # (N,5)-array, abs [cls_id,x1,y1,x2,y2]
        np.clip(canvas_boxes[:, 1:], 0, 2 * s, out=canvas_boxes[:, 1:])

    # Augment
    canvas, canvas_boxes = random_perspective(canvas, canvas_boxes,
                                              degrees=self.hyp['degrees'],
                                              translate=self.hyp['translate'],
                                              scale=self.hyp['scale'],
                                              shear=self.hyp['shear'],
                                              perspective=self.hyp['perspective'],
                                              border=[-s//2,-s//2])   # crop to [s,s]

    return canvas, canvas_boxes


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    path = self.img_files[index]
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # img: [0,255] cv_img, bgr
    # targets: abs [cls,x1,y1,x2,y2]

    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1, point_coords
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]   # x1,x2,x1,x2
        y = xy[:, [1, 3, 5, 7]]   # y1,y2,y2,y1
        xy = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], axis=1)   # x1y1x2y2

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    return img


# dynamic, not structrual, real implementation should be put into the tensor flow
def encode_boxes(anchors, boxes):
    # anchors: list of [N,2]-arr, for each grid-level
    # boxes: [N,5], normed [clsid,xc,yc,w,h]

    anchors = [np.array([[13,17],[31,25],[24,51],[61,45]]),            # stride 8
               np.array([[48,1702],[119,96],[97,189],[217,184]]),
               np.array([[171,384],[324,451],[616,618],[800,800]])]

    nL = len(anchors)
    nA = anchors[0].shape[0]
    strides = [8,16,32]
    img_size = 640
    nB = boxes.shape[0]

    off = np.array([[0,0],            # [5,2], gt grid & neighbor grids
                    [-1,0],[0,-1],[1,0],[0,1]], dtype=np.float32)   # j,k,l,m

    regress_targets = []
    cls_targets = []
    indices = []
    # for each level
    for i in range(nL):
        if nB:

            grid_h = grid_w = img_size / strides[i]
            grid_xy = np.meshgrid(np.arange(grid_w), np.arange(grid_h))   # grid indices start from axis-1

            grid_boxes = boxes.copy()
            grid_boxes[:,1:] *= [grid_w,grid_h,grid_w,grid_h]   # grid-level [xc,yc,w,h]
            # print('-------- grid boxes: -------')
            # print(grid_boxes)

            # match box to multi anchors by h/w ratio
            r = np.expand_dims(boxes[:,[3,4]]*img_size,axis=1) / anchors[i]    # [N1,N2,2]
            j = np.max(np.maximum(r, 1./r), axis=-1) < 4   # max anchor ratio, synchronize with encoding factor in predictions, valid coords in [N1,N2]
            anchor_boxes = np.tile(np.expand_dims(grid_boxes, axis=1), [1,nA,1])  # [N1,N2,5]
            anchor_id = np.tile(np.arange(nA), nB).reshape((nB,nA,1))
            anchor_boxes = np.concatenate([anchor_boxes,anchor_id],axis=-1)   # [N1,N2,6]
            anchor_boxes = anchor_boxes[j]   # filtered boxes, [N,6], [cls_id,xc,yc,w,h,anchor_id]
            # print('-------- valid boxes with multi anchors: -------')
            # print(anchor_boxes)

            # match box to multi grids
            box_xy = anchor_boxes[:,[1,2]]   # box_xcyc
            box_xyi = [grid_w,grid_h] - box_xy
            j,k = ((box_xy%1. < 0.5) & (box_xy>1.)).T    # xc<0.5 mask & yc<0.5 mask
            l,m = ((box_xyi%1. < 0.5) & (box_xyi>1.)).T   # xc>0.5 mask & yc>0.5 mask
            j = np.stack([np.ones_like(j), j, k, l, m]).T    # [N,5], pos grid mask
            anchor_boxes = np.tile(np.expand_dims(anchor_boxes, 1), [1,5,1])[j]   # [3N,6]
            # print('-------- valid boxes with multi grids: -------')
            # print(anchor_boxes)
            offsets = np.tile(np.expand_dims(off, axis=0), [j.shape[0],1,1])[j]   # [3N,2]
            # print('-------- corresponding offsets ----------')
            # print(offsets)

        else:
            anchor_boxes = boxes[0]
            offsets = 0

        a = anchor_boxes[:,-1].astype(np.int32)   # [3N,1]
        corresponding_anchors = anchors[i][a]    # [3N,2]
        grids_xy = np.floor(anchor_boxes[:,[1,2]] + offsets)   # [3N,2]
        grids_wh = anchor_boxes[:,[3,4]]
        regress_target_xy = anchor_boxes[:,[1,2]] - grids_xy                 # regress grid distance
        regress_target_wh = grids_wh * strides[i] / corresponding_anchors    # regress ratio
        indices.append([a, grid_xy])

        cls_targets.append.append(anchor_boxes[:,[0]])
        regress_targets.append(np.concatenate([regress_target_xy,regress_target_wh], axis=1))

        return cls_targets, regress_targets


def decode_preds(anchors, pred_offsets, xy_indices):
    # anchors, [N,2], gather from [b,h,w,nA,conf_dim]
    # pred_offsets, [N,4]
    # xy_indices, [N,2]
    xc_offset = np.sigmoid(pred_offsets[:,0]) * 2 - 0.5
    yc_offset = np.sigmoid(pred_offsets[:,1]) * 2 - 0.5
    w_ratio = (np.sigmoid(pred_offsets[:,2]) * 2)**2    # *scale ?
    h_ratio = (np.sigmoid(pred_offsets[:,3]) * 2)**2    # *scale ?

    boxes_xc = xy_indices[:,0] + xc_offset
    boxes_yc = xy_indices[:,1] + yc_offset

    boxes_w = anchors[:,0] * w_ratio
    boxes_h = anchors[:,1] * h_ratio

    boxes = np.stack([boxes_xc,boxes_yc,boxes_w,boxes_h], axis=1)

    return boxes


def generate_anchors(anchor_scales, anchor_ratios, strides):
    anchors = np.array(anchor_scales).reshape((-1,1)).astype(np.float32)
    anchors = np.tile(anchors, (len(anchor_ratios),2))

    factor = np.tile(np.array(anchor_ratios).reshape((-1,1)), len(anchor_scales)).reshape((-1, 1))

    anchors /= np.sqrt(factor)
    anchors[...,:1] *= factor     # anchors for a single level, N,2

    n_anchors = len(anchor_scales) * len(anchor_ratios)
    n_levels = len(strides)

    # anchors = np.tile(anchors, (1,n_levels))   # N,2L
    # stride_scale = np.tile(np.array(strides).reshape((-1,1)), (1,2)).reshape((1,-1))
    # anchors = (anchors * stride_scale).reshape((n_anchors,2,n_levels)).transpose((2,0,1)).reshape((-1,2))

    anchors_all = [anchors*i for i in strides]

    return anchors_all    # list of [N,2], wh


if __name__ == '__main__':

    from config import get_config

    cfg = get_config(None)
    dataset_train = CustomDataset(cfg)

    print('len dataset: ', len(dataset_train))

    dataloader = build_loader(cfg)

    print('len dataloader: ',len(dataloader))

    for i, data in enumerate(dataloader):
        imgs, labels, _, _ = data
        batch_img = imgs.cpu().numpy().transpose((0,2,3,1))
        batch_labels = labels.numpy()
        print(batch_img.shape, batch_labels.shape)
        for i in range(len(batch_img)):
            canvas = batch_img[i].copy()
            canvas_h, canvas_w = canvas.shape[:2]
            boxes = batch_labels[batch_labels[:,0]==i]
            for b in boxes:
                clsid, xc, yc, w, h = b[1:]
                cv2.rectangle(canvas, (int((xc-w/2.)*canvas_w), int((yc-h/2.)*canvas_h)),
                              (int((xc+w/2.)*canvas_w), int((yc+h/2.)*canvas_h)),
                              (0,0,255), 2)
            cv2.imshow('tmp', canvas)
            cv2.waitKey(0)














