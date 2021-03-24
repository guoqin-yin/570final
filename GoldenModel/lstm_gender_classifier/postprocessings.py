import torch
import torch.nn as nn
import json
import numpy as np

from torch.autograd import Variable
from pytorch_model_summary import summary

from preprocessings import get_mfccs, load_from_pickle
from GenderClassifier import GenderClassifier
from constants import NUM_MFCC, NUM_FRAMES

model = GenderClassifier()
model.load_state_dict(torch.load('model_state_dict.pkl'))
model.eval()

extracting_features = True

if extracting_features:
    print("=========== Extracting weights data ==============")
    # basic model structure and weights description
    des_file_path = "weights_data/basic_structure.txt"
    des_file = open(des_file_path, "w+")
    print("@@@ Model structure:", file=des_file)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size(), file=des_file)

    print(file=des_file)

    for param_tensor in model.state_dict():
        print("@@@ {}:".format(param_tensor), file=des_file)
        print(model.state_dict()[param_tensor], file=des_file)

    des_file.close()

    # inner layers' weights
    layer_list = list(model.state_dict().keys())
    print(layer_list)
    name_list = []
    for string in list(model.state_dict().keys()):
        name_list.append("weights_data/"+string.replace('.', '_')+".txt")
    # print(name_list)

    for idx in range(len(name_list)):
        file_name = name_list[idx]
        layer_name = layer_list[idx]

        weights_data = np.asarray(model.state_dict()[layer_name].tolist())
        dim = len(weights_data.shape)
        if dim == 3:
            with open(file_name, "w+") as outfile:
                outfile.write("# Matrix shape: {}\n".format(weights_data.shape))
                for data_slice in weights_data:
                    np.savetxt(outfile, data_slice.T, fmt='%-15.8f')
                    outfile.write("# New slice\n")
        elif dim == 2 or dim == 1:
            with open(file_name, "w+") as outfile:
                outfile.write("# Matrix shape: {}\n".format(weights_data.shape))
                np.savetxt(outfile, weights_data, fmt='%-10.8f')


model_summary_file = "model_summary.txt"
with open(model_summary_file, "w") as msf:
    print(summary(model, torch.zeros(1, NUM_MFCC, NUM_FRAMES), show_input=True), file=msf)
    print(summary(model, torch.zeros(1, NUM_MFCC, NUM_FRAMES), show_input=False, show_hierarchical=True), file=msf)

# print(summary(model, torch.zeros(1, NUM_MFCC, NUM_FRAMES), show_input=True, show_hierarchical=True))