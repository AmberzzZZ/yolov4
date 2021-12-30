### key elements in architecture
    CSP-ize back & neck
    PAN & SPP in custom FPN neck
    mish


### 精度对标
    YOLOv4-CSP  ---  eff-D3
    YOLOv4-P5   ---  eff-D5
    YOLOv4-P6   ---  eff-D6
    YOLOv4-P7   ---  eff-D7
    with faster inference speed


### input_shape
    YOLOv4-CSP   640
    YOLOv4-P5    896
    YOLOv4-P6    1280
    YOLOv4-P7    1536


### CSP block
    跟原论文的block基本一致
    分支channel减半
    with/o inside transition block: 1x1 conv


### CSPSPP
    residual path:
        1-3-1 conv
        id + pooling branches
        1-3 conv
    skip path:
        1x1 conv


### initialization

    这个一阶段模型完全由卷积和BN组成

    对 Detection Head之前的modules：没有做特别的初始化操作，只设置了参数
    - Conv:
    - BN: epsilon=1e-3, momentum=0.03
    - ReLU: inplace=True


    Detection Head的Conv weight(1,1,c_in,c_out) & bias(c_out)
    c_out = n_anchors * (cls + 1 + 4)
    - 首先将bias reshape成(n_anchors, cls+1+4)
    - 然后对最后一维的cls bias：bias[:cls] += math.log(8 / (640/s)**2), s是stride
    - 对最后一维的conf bias：bias[cls] += math.log(0.6 / (cls-0.99)), 


### loss
    classification: focal loss
    regression: giou over pred_xywh
    objectness: based on computed giou, rescale

    box encoding:
      pred_xy based on grid_xy: pxy = sigmoid(txy)*2 - 0.5
      pred_wh based on anchor_xy: pwh = (sigmoid(twh)*2) ** 2
      个人觉得，对中心点的encoding改进完以后，更接近一个基于格子中心点的正态分布，且在格子上下定点处不截断，有利于梯度
      长宽的encoding，理论上是无界的，但是实际上如果一个预测box和anchor box的形状差异巨大，与gt的iou也不会高，所以可以限幅在[0,4]之间，有利于回归


### back
    单独训了一版csp-d53，对比resnet
    * 收敛速度慢
    * 显存占用大
    * 精度？


### training stage
    train stage 1: filter the bg
    train stage 2: refinement
    实验发现，从一开始就用iou loss容易nan，而且不好收敛，
    random start阶段，conf的iou target比较less confidence，应该是全图无activation的状态，理论上应该不影响前背景快速分类，
    但是实验下来也有点问题，所以改成了先用hard label强行把前景拉高


### training details
  
    optmizer: 
    SGD(0.01,0.937) / Adam(0.001,(0.937,0.999))
    网络结构主要由conv和BN组成，其中conv包含weight，BN包含[gamma,beta]，这三种trainable param，
    我们只给conv的weight添加weight decay，BN没给，因为BN有recale，加不加没作用

    lr schedule
    1-cycle cosine: lr_min=0.2, cos_scale=0.8(max=1.)

    tricks
    EMA
    syncBN


### data
  
    model strides: output levels, 如P5就是[8,16,32]，P6是[8,16,32,64]
    网络的输入(image的长宽)必须是strides的倍数，


    rectangular training & square training
    输入图像要等比例缩放的
    常规的square training就是网络始终输入正方形，图片的长边缩放到目标尺寸，短边padding到目标尺寸
    rectangular training则是图片的长边缩放到目标尺寸，短边padding到strides的倍数，save memory
    rectangular方法在数据预处理阶段要先根据长宽比聚类，比例相似的放在一个batch，然后取batch中的Hmax和Wmax作为batch input size

    mosiac：
    当前图片和全量中任意三张图片

    mixup：
    beta(0.8,0.8)，只出现在mosaic分支里面，只对两个mosaic图片做了mixup

    random_perspective:
    常规形变，rotation/translate/scale/shear

    augment_hsv
    常规色彩变换

    flip


## model architecture
    
    model.yaml里面主要包含backbone和head(fpn+detection head)
    保存为list，每个元素为[from, number, module, args]
    - from是绝对idx，当前module的输入的来源
    - number是当前module堆叠的次数
    - module和args用于构建当前module的结构

    这些modules定义在models/common.py里面：
    --- backbone ----
    Conv(c_out, kernel=1, strides=1, pad=None, groups=1, act=True): conv-bn-Mish
    Bottleneck(c_out, shortcut=True, groups=1, expand_ratio=0.5): standard bottleneck, conv-conv with shortcut, 注意是CSP paper定义的standard bottleneck，也就是只有两层卷积，而不是resnet那个3层卷积的版本
    BottleneckCSP(c_out, n_blocks=1, shortcut=True, groups=1, expand_ratio=0.5): CSP block 
    ----- fpn -----
    SPPCSP(c_out, n_blocks=1, shortcut=False, groups=1, expand_ratio=0.5, k=(5, 9, 13)): CSP block的residual path里面再加上SPP，多尺度pooling
    nn.Upsample(size=None,factor=None,mode='nearest'): 上采样
    Concat(dimension=1): merge by dim
    BottleneckCSP2:(c_out, n_blocks=1, shortcut=False, groups=1, expand_ratio=0.5): CSP block
    ---- head -----
    Detect(n_cls=80, anchors=(), ch=())



    以yolov4-p5为例：
    
    backbone:
      # [from, number, module, args]
      [[-1, 1, Conv, [32, 3, 1]],  # 0
       [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
       [-1, 1, BottleneckCSP, [64]],
       [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4
       [-1, 3, BottleneckCSP, [128]],
       [-1, 1, Conv, [256, 3, 2]],  # 5-P3/8
       [-1, 15, BottleneckCSP, [256]],
       [-1, 1, Conv, [512, 3, 2]],  # 7-P4/16
       [-1, 15, BottleneckCSP, [512]],
       [-1, 1, Conv, [1024, 3, 2]], # 9-P5/32
       [-1, 7, BottleneckCSP, [1024]],  # 10
      ]

    Input: (b,H,W,3)

    # stem: module 0-1
    Conv1: (b,H,W,32)
    Conv2: (b,H/2,W/2,64)    

    # stage1: module 2-3
    BottleneckCSP: (b,H/2,W/2,64)    ----> stride2, P1
    downSamp Conv: (b,H/4,W/4,128)     

    # stage2: module 4-5
    3xBottleneckCSP: (b,H/4,W/4,128)   ----> stride4, P2
    downSamp Conv: (b,H/8,W/8,256)     

    # stage3: module 6-7
    15xBottleneckCSP: (b,H/8,W/8,256)     ----> stride8, P3
    downSamp Conv: (b,H/16,W/16,512)     

    # stage4: module 8-9
    15xBottleneckCSP: (b,H/16,W/16,512)    ----> stride16, P4
    downSamp Conv: (b,H/32,W/32,1024)     

    # stage5: module 10
    7xBottleneckCSP: (b,H/32,W/32,1024)    ----> stride32, P5


    fpn:
      [[-1, 1, SPPCSP, [512]], # 11
       [-1, 1, Conv, [256, 1, 1]],
       [-1, 1, nn.Upsample, [None, 2, 'nearest']],
       [8, 1, Conv, [256, 1, 1]], # route backbone P4
       [[-1, -2], 1, Concat, [1]],
       [-1, 3, BottleneckCSP2, [256]], # 16 
       [-1, 1, Conv, [128, 1, 1]],
       [-1, 1, nn.Upsample, [None, 2, 'nearest']],
       [6, 1, Conv, [128, 1, 1]], # route backbone P3
       [[-1, -2], 1, Concat, [1]],
       [-1, 3, BottleneckCSP2, [128]], # 21
       [-1, 1, Conv, [256, 3, 1]],
       [-2, 1, Conv, [256, 3, 2]],
       [[-1, 16], 1, Concat, [1]],  # cat
       [-1, 3, BottleneckCSP2, [256]], # 25
       [-1, 1, Conv, [512, 3, 1]],
       [-2, 1, Conv, [512, 3, 2]],
       [[-1, 11], 1, Concat, [1]],  # cat
       [-1, 3, BottleneckCSP2, [512]], # 29
       [-1, 1, Conv, [1024, 3, 1]],
      ]


    P3(256)    ------    conv(128)  ---------   cat(256) - 3xCSP(128)  - conv(256)
                                                   /          |
                                            ConvUP(128)   s2conv(256)
                                                 /            |
    P4(512)  ---  conv(256)  -- cat(512) - 3xCSP(256)    -   cat(512) - 3xCSP(256)  - conv(512)
                                 /                                        |
                            ConvUp(256)                                s2conv(512)
                               /                                          |
    P5(1024)  ------  SPPCSP(512)        -------------------------      cat(1024)  - 3xCSP(512)  - conv(1024)



    detetion head:
    [[22,26,30], 1, Detect, [n_cls, anchors]]

    Detect([P3,P4,P5])
    就是一层卷积：1x1 conv with bias + sigmoid, (b,h,w,a,c+1+4)

    box_offsets
        xy_offset: y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
        wh_offset: y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
    * wh这个好理解，sigmoid出来[0,1]，*2**2就变换到[0,4]了，可以对anchor box缩放/扩大
    * xy这个，sigmoid出来[0,1]，*2-0.5变换到[-0.5,1.5]，这个作为grid center的偏移量，相比较于原始的[0,1]激活区间更大一点，有助于偏移量比较大的object的收敛
    * ref: https://github.com/WongKinYiu/ScaledYOLOv4/issues/90#
    * ref_origin: https://github.com/AlexeyAB/darknet/issues/3293














