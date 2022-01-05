# config file

import os
import yaml
from yacs.config import CfgNode as CN
import numpy as np


_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 2
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = 'data/coco/tinyval2014/'
_C.DATA.LABEL_PATH = 'data/coco/txt/'
# Dataset name
_C.DATA.DATASET = 'COCO'   # ImageNet
# Input image size
_C.DATA.IMG_SIZE = 640
_C.DATA.IMG_CHANNELS = 3
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Anchor settings
# -----------------------------------------------------------------------------
_C.ANCHOR = CN()
_C.ANCHOR.SCALE = [8]               # anchor scale, relative to grid strides
_C.ANCHOR.RATIO = [0.5, 1.0, 2.0]   # anchor h/w ratio
_C.ANCHOR.STRIDES = [8,16,32]       # grid strides

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 13
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings for classification
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
# _C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Augmentation settings for object detection
# -----------------------------------------------------------------------------
_C.AUG.MIXUP = 0.0
_C.AUG.HSV = [0.015,0.7,0.4]
_C.AUG.DEGREE = 0.
_C.AUG.TRANSLATE = 0.1     # 0-0.5
_C.AUG.SCALE = 0.5
_C.AUG.SHEAR = 0.
_C.AUG.PERSPECTIVE = 0.001    # 0-0.001
_C.AUG.FLIPUD = 0.
_C.AUG.FLIPLR = 0.5

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True


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


scales = _C.ANCHOR.SCALE
ratios = _C.ANCHOR.RATIO
strides = _C.ANCHOR.STRIDES
_C.ANCHOR.ANCHORS = generate_anchors(scales, ratios, strides)  # list of [NA,2]
_C.ANCHOR.NA = len(scales) * len(ratios)   # n_anchors per level


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if args:
        _update_config_from_file(config, args.cfg)

        config.defrost()
        if args.opts:
            config.merge_from_list(args.opts)

        # merge from specific arguments
        if args.batch_size:
            config.DATA.BATCH_SIZE = args.batch_size
        if args.data_path:
            config.DATA.DATA_PATH = args.data_path
        if args.zip:
            config.DATA.ZIP_MODE = True
        if args.cache_mode:
            config.DATA.CACHE_MODE = args.cache_mode
        if args.resume:
            config.MODEL.RESUME = args.resume
        if args.accumulation_steps:
            config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
        if args.use_checkpoint:
            config.TRAIN.USE_CHECKPOINT = True
        if args.amp_opt_level:
            config.AMP_OPT_LEVEL = args.amp_opt_level
        if args.output:
            config.OUTPUT = args.output
        if args.tag:
            config.TAG = args.tag
        if args.eval:
            config.EVAL_MODE = True
        if args.throughput:
            config.THROUGHPUT_MODE = True

        # set local rank for distributed training
        config.LOCAL_RANK = args.local_rank

        # output folder
        config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

        config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


if __name__ == '__main__':

    cfg = get_config(None)

    print(cfg.ANCHOR.ANCHORS)
    print(cfg.ANCHOR.NA)



