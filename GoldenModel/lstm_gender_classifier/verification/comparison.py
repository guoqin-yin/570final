import sys
sys.path.append('../')

import os
import torch
import numpy as np

from torch.autograd import Variable
from GenderClassifier import GenderClassifier
from preprocessings import get_mfccs, load_from_pickle, regularize_data
from args import Args
from constants import NUM_MFCC, NUM_FRAMES

from fwdfunctions import sigmoid, softmax, get_conv_out, get_fc_out, get_lstm_out, get_pool_out

import matplotlib.pyplot as plt

#=======================================================
# Load Model & weights extraction
#=======================================================
model = GenderClassifier()
model.load_state_dict(torch.load('../model_state_dict.pkl'))
model.eval()

layer_list = list(model.state_dict().keys())
print(layer_list)

conv1_weights   = model.state_dict()[layer_list[0]].detach().numpy()
lstm_weight_ih  = model.state_dict()[layer_list[1]].detach().numpy()
lstm_weight_hh  = model.state_dict()[layer_list[2]].detach().numpy()
fc1_weight  = model.state_dict()[layer_list[3]].detach().numpy()

args = Args()

#=======================================================
# Load Data
#=======================================================
x_test_mfccs   = np.asarray(get_mfccs(pickle_file="X-test-mfccs.pkl"))
y_test         = load_from_pickle(filename="y-test.pkl")
x_test_mfccs = regularize_data(x_test_mfccs)
x_test_tensor  = Variable(torch.Tensor(x_test_mfccs), requires_grad=False)
y_test_tensor  = Variable(torch.Tensor(y_test), requires_grad=False)


x_data_tensor   = x_test_tensor
y_data_tensor   = y_test_tensor
x_data          = x_data_tensor.numpy()
y_data          = y_data_tensor.numpy()

#=======================================================
# Accuracy of valid data
#=======================================================
print("------------- Test Accuracy -------------")

outputs = model(x_data_tensor)
_, outputs_label = outputs.max(dim=1)
accuracy = int(sum(outputs_label == y_data_tensor))/len(y_data_tensor)
print("data size: {}, accuracy: {:.5f}%".format(len(y_data_tensor), 100*accuracy))

#=======================================================
# Forward Calculation: before quantization
#=======================================================
print("------------- Before Quantization -------------")

test_prev_quantization = False

# forwarding data
def forward_data(x_data):
    conv1out    = get_conv_out(conv1_weights, x_data, 4)
    lstm_in     = np.transpose(pooledout, (0, 2, 1))
    lstmout     = get_lstm_out(lstm_in, lstm_weight_ih, lstm_weight_hh, hidden_size=16)  
    fcin        = lstmout.reshape(lstmout.shape[0], -1)
    fc1out      = get_fc_out(fcin, fc1_weight)
    # fc2out      = get_fc_out(fc1out, fc2_weight, fc2_bias)
    # fc3out      = get_fc_out(fc2out, fc3_weight, fc3_bias)

    return fc1out

if test_prev_quantization:
    fcout = forward_data(x_data)
    classres = np.zeros((fcout.shape[0]))
    for i in range(fcout.shape[0]):    
        classres[i] = np.argmax(softmax(fcout[i]))
    accuracy = int(sum(classres == y_data))/len(y_data)
    print("data size: {}, accuracy: {:.5f}%".format(len(y_data), 100*accuracy))


# EXTRACT correct output data

EXTRACT_INNER_DATA = False

if EXTRACT_INNER_DATA:

    data = x_test_tensor.numpy()[0,:,:]
    data = np.expand_dims(data, axis=0)
    # print(data.shape)
    conv1out    = get_conv_out(conv1_weights, data, 4)
    conv2out    = get_conv_out(conv2_weights, conv1out, 2)

    conv1out = conv1out.astype(int)
    conv2out = regularize_data(conv2out.astype(int))
    pooledout   = get_pool_out(conv2out, kernel_size=16)
    lstm_in     = np.transpose(pooledout, (0, 2, 1))
    lstmout     = get_lstm_out(lstm_in, lstm_weight_ih, lstm_weight_hh, hidden_size=16, extract_data=True)

    # print(conv2out.shape)

    # file_name = "conv1out.txt"
    # with open(file_name, "w+") as outfile:
    #     outfile.write("# Matrix shape: {}\n".format(conv1out.shape))
    #     np.savetxt(outfile, conv1out[0,:,:].T, fmt='%10d')

    # print(pooledout.shape)

    conv2out = pooledout

    conv2out = conv2out[0,:,:].T
    file_name = "conv2out.txt"
    with open(file_name, "w+") as outfile:
        outfile.write("# Matrix shape: {}\n".format(conv2out.shape))
        for i in range(conv2out.shape[0]):
            line = conv2out[i].tolist()
            line = [str(x) for x in line]
            outfile.write(" ".join(line[::-1]))
            outfile.write("\n")
            # np.savetxt(outfile, conv2out[0,:,:].T, fmt='%4d')



#=======================================================
# Forward Calculation: after quantization
#=======================================================
print("------------- After Quantization -------------")

test_quantization = True

if test_quantization:

    name_list = []
    for string in list(model.state_dict().keys()):
        name_list.append("../quantization/"+string.replace('.', '_')+".txt")

    conv1_weights  = np.loadtxt(name_list[0])
    conv1_weights  = conv1_weights.reshape(8, 4, 16)
    conv1_weights  = np.transpose(conv1_weights, (0, 2, 1))
    conv2_weights  = np.loadtxt(name_list[1])
    conv2_weights  = conv2_weights.reshape(4, 2, 8)
    conv2_weights  = np.transpose(conv2_weights, (0, 2, 1))

    lstm_weight_ih = np.loadtxt(name_list[2])
    lstm_weight_hh = np.loadtxt(name_list[3])
    # lstm_bias_ih   = np.loadtxt(name_list[4])
    # lstm_bias_hh   = np.loadtxt(name_list[5])

    fc1_weight     = np.loadtxt(name_list[4])
    fc1_bias       = np.loadtxt(name_list[5])
    # fc2_weight     = np.loadtxt(name_list[8])
    # fc2_bias       = np.loadtxt(name_list[9])

    fcout  = forward_data(x_data)
    classres = np.zeros((fcout.shape[0]))
    for i in range(fcout.shape[0]):    
        classres[i] = np.argmax(softmax(fcout[i]))
    accuracy = int(sum(classres == y_data))/len(y_data)
    print("data size: {}, accuracy: {:.5f}%".format(len(y_data), 100*accuracy))

