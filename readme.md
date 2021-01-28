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





      






