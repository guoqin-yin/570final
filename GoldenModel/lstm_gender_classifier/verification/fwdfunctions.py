import sys
sys.path.append('../')

import torch
import numpy as np

from torch.autograd import Variable
from GenderClassifier import GenderClassifier
from preprocessings import get_mfccs, load_from_pickle
from args import Args

# #=======================================================
# # Load Model & weights extraction
# #=======================================================
# model = GenderClassifier()
# model.load_state_dict(torch.load('../model_state_dict.pkl'))
# model.eval()

# layer_list = list(model.state_dict().keys())

# #=======================================================
# # Load Data
# #=======================================================
# x_valid_mfccs = get_mfccs(pickle_file="X-valid-mfccs.pkl")
# y_valid = load_from_pickle(filename="y-valid.pkl")
# x_valid_tensor = Variable(torch.Tensor(x_valid_mfccs), requires_grad=False)
# y_valid_tensor = Variable(torch.LongTensor(y_valid), requires_grad=False)

# args = Args()

# #=======================================================
# # Accuracy of valid data
# #=======================================================
# print("------------- Test Accuracy -------------")

# outputs = model(x_valid_tensor)
# _, outputs_label = outputs.max(dim=1)
# accuracy = int(sum(outputs_label == y_valid_tensor))/len(y_valid_tensor)
# print("data size: {}, accuracy: {:.3f}%".format(len(y_valid), 100*accuracy))


# #=======================================================
# # Forward Calculation
# #=======================================================
# print("------------- Before Quantization -------------")
# # print(layer_list)

# m_c1out = model.get_conv1_out(x_valid_tensor)
# m_cout, m_lstmout = model.get_lstm_out(x_valid_tensor)
# # print(c1out.detach().numpy()[1,:,0])

# conv1_weights   = model.state_dict()[layer_list[0]].detach().numpy()
# conv1_bias      = model.state_dict()[layer_list[1]].detach().numpy()
# conv2_weights   = model.state_dict()[layer_list[2]].detach().numpy()
# conv2_bias      = model.state_dict()[layer_list[3]].detach().numpy()

# lstm_weight_ih  = model.state_dict()[layer_list[4]].detach().numpy()
# lstm_weight_hh  = model.state_dict()[layer_list[5]].detach().numpy()
# lstm_bias_ih    = model.state_dict()[layer_list[6]].detach().numpy()
# lstm_bias_hh    = model.state_dict()[layer_list[7]].detach().numpy()

# x_data          = x_valid_tensor.numpy()

def get_conv_out(conv_weights, x_data, kernel_size):
    """
    conv_weights_dim    [NUM_out_channel, NUM_in_channel, kernel_size]
    conv_bias_dim       [NUM_out_channel,]
    x_data              [NUM_SAMPLES, NUM_MFCC, NUM_FRAMES]
    """
    res = np.zeros((x_data.shape[0], conv_weights.shape[0], x_data.shape[2]-kernel_size+1))
    for sidx in range(res.shape[0]):
        for i in range(res.shape[2]):
            convout = np.dot(conv_weights.reshape(conv_weights.shape[0], -1), x_data[sidx,:,i:i+kernel_size].flatten())
            # for k in range(convout.shape[0]):
            #     if convout[k] <= 0:
            #         convout[k] = 0
            res[sidx,:,i] = convout
    return res

def get_pool_out(fwd_data, kernel_size):
    """
    fwd_data_dim    [NUM_SAMPLES, NUM_channel, NUM_FRAMES]
    pooled_data_dim [NUM_SAMPLES, NUM_channel, NUM_FRAMES/kernel_size]
    """
    res = np.zeros((fwd_data.shape[0], fwd_data.shape[1], int(fwd_data.shape[2]/kernel_size)))
    for i in range(int(fwd_data.shape[2]/kernel_size)):
        res[:,:,i] = np.amax(fwd_data[:,:,i*kernel_size:i*kernel_size+kernel_size],2)

    return res


def sigmoid(x):
    # return 1./(1+np.exp(-x))
    
    bd = 2
    res = np.zeros(x.shape[0])
    for idx in range(x.shape[0]):
        if x[idx] > bd:
            res[idx] = 1
        elif x[idx] < -bd:
            res[idx] = 0
        else:
            res[idx] = x[idx]/4+0.5
    return res

def tanh(x):
    # return np.sinh(x)/np.cosh(x)


    bd = 1
    res = np.zeros(x.shape[0])
    for idx in range(x.shape[0]):
        if x[idx] > bd:
            res[idx] = 1
        elif x[idx] < -bd:
            res[idx] = 0
        else:
            res[idx] = x[idx]
    return res

def sigmoid2(x):
    res = np.zeros_like(x)
    for iter in range(x.shape[0]):
        val = x[iter]
        temp = 0
        if val <= -6:
            temp = 0
        elif val > -6 and val <= -3:
            temp = 0.203125+0.0703125*val+0*val*val
        elif val > -3 and val <= 0:
            temp = 0.5+0.265625*val+0.0390625*val*val
        elif val > 0 and val <= 3:
            temp = 0.4921875+0.265625*val-0.0390625**val*val
        elif val > 3 and val <= 6:
            temp = 0.7890625+0.0703125*val-0.0078125*val*val
        else:
            temp = 1
        res[iter] = temp
    
    return res

def tanh2(x):
    res = np.zeros_like(x)
    for iter in range(x.shape[0]):
        val = x[iter]
        temp = 0
        if val <= -3:
            temp = -1
        elif val > -3 and val <= -1:
            temp = -0.84375+0.4609375*val+0.0859375*val*val
        elif val > -1 and val <= 0:
            temp =  0+1.078125*val+0.3125*val*val
        elif val > 0 and val <= 1:
            temp = 0+1.78125*val-0.3203125*val*val
        elif val > 1 and val <= 3:
            temp = 0.3984375+0.4609375*val-0.09375*val*val
        else:
            temp = 1
        res[iter] = temp
    
    return res


# def sigmoid2(x):
#     res = np.zeros_like(x)
#     for iter in range(x.shape[0]):
#         val = x[iter]
#         temp = 0
#         if val <= -6:
#             temp = 0
#         elif val > -6 and val <= -3:
#             temp = 0.20323428+0.0717631*val+0.00642858*val*val
#         elif val > -3 and val <= 0:
#             temp = 0.50195831+0.27269294*val+0.04059181*val*val
#         elif val > 0 and val <= 3:
#             temp = 0.49805785+0.27266221*val-0.04058115**val*val
#         elif val > 3 and val <= 6:
#             temp = 0.7967568+0.07175359*val-0.00642671*val*val
#         else:
#             temp = 1
#         res[iter] = temp
    
#     return res

# def tanh2(x):
#     res = np.zeros_like(x)
#     for iter in range(x.shape[0]):
#         val = x[iter]
#         temp = 0
#         if val <= -3:
#             temp = -1
#         elif val > -3 and val <= -1:
#             temp = -0.39814608+0.46527859*val+0.09007576*val*val
#         elif val > -1 and val <= 0:
#             temp =  0.0031444+1.08381219*val+0.31592922*val*val
#         elif val > 0 and val <= 1:
#             temp = -0.00349517+1.08538355*val-0.31676793*val*val
#         elif val > 1 and val <= 3:
#             temp = 0.39878032+0.46509003*val-0.09013554*val*val
#         else:
#             temp = 1
#         res[iter] = temp
    
#     return res


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def get_lstm_out(fwd_data, weight_ih, weight_hh, hidden_size, extract_data=False):
    """
    fwd_data_dim    [NUM_SAMPLES, pooled_NUM_FRAMES(seq_len), NUM_channel]
    outdata_dim     [NUM_SAMPLES, pooled_NUM_FRAMES(seq_len), hidden_size]
    (W_ii|W_if|W_ig|W_io)
    """
    res = np.zeros((fwd_data.shape[0], fwd_data.shape[1], hidden_size))
    # input_size = fwd_data.shape[2]
    # time_steps = fwd_data.shape[1]
    # print("res shape:", res.shape)

    if extract_data:
        f = open("lstm_inner_data.txt", "w+")
    
    for sidx in range(res.shape[0]):
        h_state = np.zeros((hidden_size))
        cell_state = np.zeros((hidden_size))
        
        for t in range(fwd_data.shape[1]):

            step_i_temp = np.dot(weight_ih[0:hidden_size,:], fwd_data[sidx,t,:])+np.dot(weight_hh[0:hidden_size,:], h_state)
            step_i = sigmoid(step_i_temp)

            step_f_temp = np.dot(weight_ih[hidden_size:2*hidden_size,:], fwd_data[sidx,t,:])+np.dot(weight_hh[hidden_size:2*hidden_size,:], h_state)
            step_f = sigmoid(step_f_temp)

            step_g_temp = np.dot(weight_ih[2*hidden_size:3*hidden_size,:], fwd_data[sidx,t,:])+np.dot(weight_hh[2*hidden_size:3*hidden_size,:], h_state)
            step_g = tanh(step_g_temp)
            
            step_o_temp = np.dot(weight_ih[3*hidden_size:4*hidden_size,:], fwd_data[sidx,t,:])+np.dot(weight_hh[3*hidden_size:4*hidden_size,:], h_state)
            step_o = sigmoid(step_o_temp)

            cell_state = step_g * step_i + cell_state * step_f
            h_state = step_o * tanh(cell_state)

            if extract_data and sidx == 0:
                if t == 0:
                    print(weight_ih[0:hidden_size,:])
                    print(fwd_data[sidx,t,:])
                    print(weight_hh[0:hidden_size,:])
                    print(h_state)
                f.write("time {}\ni:{}\nacti_i:{}\nf:{}\nacti_f:{}\nc:{}\nacti_c:{}\no:{}\nacti_o:{}\ncell_state:{}\nh_state:{}\n".format(t, step_i_temp, step_i, step_f_temp, step_f, step_g_temp, step_g, step_o_temp, step_o, cell_state, h_state))

            res[sidx,t,:] = h_state
    # print("max:{}, min:{}".format(vmax, vmin))
    return res


def get_fc_out(fwd_data, fc_weight):
    fcout = np.dot(fwd_data, fc_weight.T)
    return fcout