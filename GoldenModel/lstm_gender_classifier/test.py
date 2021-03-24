# path : /afs/umich.edu/class/eecs627/w20/group5/GoldenModel/lstm_gender_classifier
# source venv/bin/activate

import torch
import torch.nn as nn

import numpy as np
from preprocessings import get_mfccs, load_from_pickle

import matplotlib.pyplot as plt

x_train_mfccs = np.asarray(get_mfccs(pickle_file="X-train-mfccs.pkl"))
x_test_mfccs = np.asarray(get_mfccs(pickle_file="X-test-mfccs.pkl"))

y_train = load_from_pickle(filename="y-train.pkl")
y_test = load_from_pickle(filename="y-test.pkl")

input_data = open("./input_data/input_data.txt",'w')

x_train_mfccs = x_train_mfccs.astype(int)
x_test_mfccs = x_test_mfccs.astype(int)

Nx_train = x_train_mfccs.shape[0]
Ny_train = x_train_mfccs.shape[1]
Nz_train = x_train_mfccs.shape[2]
for i in range (0,Nx_train):
    for j in range (0,Ny_train):
        for k in range (0,Nz_train):
            if (x_train_mfccs[i,j,k] > 127):
                x_train_mfccs[i,j,k] = 127
            elif (x_train_mfccs[i,j,k] < -128):
                x_train_mfccs[i,j,k] = -128

Nx_test = x_test_mfccs.shape[0]
Ny_test = x_test_mfccs.shape[1]
Nz_test = x_test_mfccs.shape[2]
for i in range (0,Nx_test):
    for j in range (0,Ny_test):
        for k in range (0,Nz_test):
            if (x_test_mfccs[i,j,k] > 127):
                x_test_mfccs[i,j,k] = 127
            elif (x_test_mfccs[i,j,k] < -128):
                x_test_mfccs[i,j,k] = -128

print (Nx_test,Ny_test,Nz_test)

np.savetxt(input_data, x_test_mfccs[0,:,:].T, fmt='%-4d')

# for i in range (0,Ny_test):
#     for j in range (0,Nz_test):
#         input_data.write(str(x_test_mfccs[0,i,j]))
#         input_data.write("  ")
#     input_data.write("\n")
# input_data.write(x_test_mfccs[0,:,:])

# data = x_test_mfccs.reshape(-1)
# plt.hist(data)
# plt.savefig("dist.png")




# print("max:{}, min:{}".format(np.max(x_train_mfccs), np.min(x_train_mfccs)))
# print(np.asarray(x_train_mfccs).shape)

# data = np.asarray(x_train_mfccs).reshape(-1)
# plt.hist(data)
# plt.savefig('dist.png')

# bd = -700
# count = 0
# for sample in range(len(x_train_mfccs)):
#     for channel in range(len(x_train_mfccs[sample])):
#         for i in range(len(x_train_mfccs[sample][channel])):
#             if x_train_mfccs[sample][channel][i] < bd:
#                 count += 1

# print(count)
# x_train_mfccs = np.tanh(x_train_mfccs)
# # print(x_train_mfccs[0,:,0])

# x_train_tensor = Variable(torch.Tensor(x_train_mfccs), requires_grad=False)
# # print(x_train_tensor.size())

# conv1_weights   = np.loadtxt("weights_data/conv_0_weights.txt")
# print(conv1_weights.shape)