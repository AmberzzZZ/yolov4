import h5py
import numpy as np
import torch
import sys
import yaml
sys.path.append("models/")
sys.path.append("utils/")
sys.path.append("../")


with open("models/yolov4-p5.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
data = data['backbone'] + data['head']
data = [i for i in data if 'nn.Upsample' not in i]
for idx, d in enumerate(data):
    print(idx, d)


torch_model = torch.load("../weights/yolov4-p5.pt", map_location='cpu')
model = torch_model['model']
sequential = list(model.children())[0]
sequential = [i for i in sequential if 'Upsample' not in i.type]

# for sub_module in sequential.children():
#     print('--------------- sub module : ', sub_module.type)
#     print(sub_module)


from yolov4 import yolov4
keras_model = yolov4(input_shape=(896,896,3), n_classes=80, n_anchors=3)

idx = -1
fpn_outs = []
cat = 0
for layer in keras_model.layers[1:-6]:

    if 'samp' in layer.name:
        continue

    idx += 1
    torch_idx = idx + len(fpn_outs)
    print(idx, layer.name, sequential[torch_idx].type, data[torch_idx])

    if 'cat' in layer.name:
        cat += 1

    torch_weights = [[k,v] for k,v in sequential[torch_idx].state_dict().items() if 'weight' in k or 'bias' in k or 'running' in k]
    torch_weights = [np.transpose(v, (2,3,1,0)) if len(v.shape)==4 else v for k,v in torch_weights]

    if cat >= 2 and 'Conv' in sequential[torch_idx].type and data[torch_idx][-1][-1]== 1:
        fpn_outs.append(torch_weights)
        continue

    if not layer.get_weights():
        continue

    cnt = 0
    for l in layer.layers:
        n_params = len(l.get_weights())
        l.set_weights(torch_weights[cnt:cnt+n_params])
        cnt += n_params

# last torch conv
torch_idx += 1
torch_weights = [[k,v] for k,v in sequential[torch_idx].state_dict().items() if 'weight' in k or 'bias' in k or 'running' in k]
torch_weights = [np.transpose(v, (2,3,1,0)) if len(v.shape)==4 else v for k,v in torch_weights]
fpn_outs.append(torch_weights)


# keras fpn outs
for idx, layer in enumerate(keras_model.layers[-6:-3]):
    cnt = 0
    for l in layer.layers:
        n_params = len(l.get_weights())
        l.set_weights(fpn_outs[idx][cnt:cnt+n_params])
        cnt += n_params

keras_model.save_weights("yolov4-p5.h5")








