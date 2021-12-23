import h5py
import numpy as np
import torch
import torch.nn as nn
import sys
import yaml
sys.path.append("models/")
sys.path.append("utiis/")


# torch order:
# conv: weight & bias, [out,in,k,k], [out]
# bn: gamma, beta, mean, var, [out]

torch_model = torch.load("weights/yolov4-p5.pt", map_location='cpu')
with open("models/yolov4-p5.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
data = data['backbone'] + data['head']
data = [i for i in data if 'nn.Upsample' not in i]

print(torch_model.keys())
# dict_keys(['epoch', 'best_fitness', 'training_results', 'model', 'optimizer', 'wandb_id'])

model = torch_model['model']     # model object

sequential = list(model.children())[0]
sequential = [i for i in sequential if 'Upsample' not in i.type]

# for idx,sub_module in enumerate(sequential):
#     print('---------', idx, sub_module.type, '---------')
#     weights = sub_module.state_dict()
#     print([v.shape for k,v in weights.items() if 'weight' in k or 'bias' in k or 'running' in k])

# weights = model.state_dict()
# print(weights.keys())



from yolov4 import yolov4

keras_model = yolov4(input_shape=(896,896,3))
idx = -1
cat = 0
output_convs = []
for layer in keras_model.layers[:-6]:
    if 'input' in layer.name or 'samp' in layer.name:
        continue

    idx += 1
    print(idx, layer.name, sequential[idx+len(output_convs)].type)

    if 'cat' in layer.name:
        cat += 1

    if not layer.get_weights():
        continue

    # if idx<11:     # for quick debug
    #     continue

    torch_weights = sequential[idx+len(output_convs)].state_dict()
    torch_weights = {k:v for k,v in torch_weights.items() if 'weight' in k or 'bias' in k or 'running' in k}
    keras_weights = [np.transpose(v, (2,3,1,0)) if len(v.shape)==4 else v for k,v in torch_weights.items()]
    # print([tuple(i.shape) for i in keras_weights])
    print(data[idx+len(output_convs)])
    if cat>=2 and 'Conv' in sequential[idx+len(output_convs)].type and data[idx+len(output_convs)][3][-1]==1:  # stride1 conv
        print("save the fpn output conv")
        output_convs.append(keras_weights)
        continue

    print('torch: ', [[k,tuple(v.shape)] for k,v in torch_weights.items()], len(torch_weights.items()))
    print('keras: ', [i.shape for i in layer.get_weights()], len(layer.get_weights()))
    # layer.set_weights()    not exactly the same order

    if not layer.get_weights():
        continue

    cnt = 0
    for l in layer.layers:
        n_params = len(l.get_weights())
        l.set_weights(keras_weights[cnt:cnt+n_params])
        cnt += n_params

# last torch conv
torch_weights = sequential[idx+3].state_dict()
torch_weights = {k:v for k,v in torch_weights.items() if 'weight' in k or 'bias' in k or 'running' in k}
keras_weights = [np.transpose(v, (2,3,1,0)) if len(v.shape)==4 else v for k,v in torch_weights.items()]
output_convs.append(keras_weights)

# mismatch order in fpn_outputs
for idx, layer in enumerate(keras_model.layers[-6:-3]):   # from the P5 level
    print(layer.name)
    print(len(output_convs[idx]))

    cnt = 0
    for l in layer.layers:
        n_params = len(l.get_weights())
        l.set_weights(output_convs[idx][cnt:cnt+n_params])
        cnt += n_params


keras_model.save_weights("weights/yolov4-p5.h5")













