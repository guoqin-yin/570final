import torch
import torch.nn as nn
import numpy as np

def sigmoid(x):
   return 1./(1+np.exp(-x))

def get_lstm_out(fwd_data, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
    """
    fwd_data_dim    [NUM_SAMPLES, pooled_NUM_FRAMES(seq_len), NUM_channel]
    outdata_dim     [NUM_SAMPLES, pooled_NUM_FRAMES(seq_len), hidden_size]
    (W_ii|W_if|W_ig|W_io)
    """
    res = np.zeros((fwd_data.shape[0], fwd_data.shape[1], hidden_size))
    input_size = fwd_data.shape[2]
    time_steps = fwd_data.shape[1]
    print(res.shape[0], fwd_data.shape[1])
    for sidx in range(res.shape[0]):
        h_state = np.zeros((hidden_size))
        cell_state = np.zeros((hidden_size))
        for t in range(fwd_data.shape[1]):
            step_i = sigmoid(np.dot(weight_ih[0:hidden_size,:], fwd_data[sidx,t,:])+bias_ih[0:hidden_size]+np.dot(weight_hh[0:hidden_size,:], h_state) + bias_hh[0:hidden_size])

            step_f = sigmoid(np.dot(weight_ih[hidden_size:2*hidden_size,:], fwd_data[sidx,t,:])+bias_ih[hidden_size:2*hidden_size]+np.dot(weight_hh[hidden_size:2*hidden_size,:], h_state) + bias_hh[hidden_size:2*hidden_size])

            step_g = np.tanh(np.dot(weight_ih[2*hidden_size:3*hidden_size,:], fwd_data[sidx,t,:])+bias_ih[2*hidden_size:3*hidden_size]+np.dot(weight_hh[2*hidden_size:3*hidden_size,:], h_state) + bias_hh[2*hidden_size:3*hidden_size])
            
            step_o = sigmoid(np.dot(weight_ih[3*hidden_size:4*hidden_size,:], fwd_data[sidx,t,:])+bias_ih[3*hidden_size:4*hidden_size]+np.dot(weight_hh[3*hidden_size:4*hidden_size,:], h_state) + bias_hh[3*hidden_size:4*hidden_size])

            cell_state = step_g * step_i + cell_state * step_f
            h_state = step_o * np.tanh(cell_state)

            res[sidx,t,:] = h_state

    return res, h_state

h0 = torch.zeros(1, 10, 16)
c0 = torch.zeros(1, 10, 16)


rnn = nn.LSTM(8, 16, batch_first=True)
input = torch.randn(10, 5, 8)
output, (hn, cn) = rnn(input, (h0, c0))
layer_list = list(rnn.state_dict().keys())
print("============ output ===============")
print(output.shape)
print(output[5,:,0].detach().numpy())

res, h_state = get_lstm_out(input, rnn.state_dict()[layer_list[0]].detach().numpy(), rnn.state_dict()[layer_list[1]].detach().numpy(), rnn.state_dict()[layer_list[2]].detach().numpy(), rnn.state_dict()[layer_list[3]].detach().numpy(), 16)

print(res[5,:,0])

# print("============ hstate ===============")
# print(hn[0,0,:].detach().numpy())
# print(h_state)