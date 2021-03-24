import numpy as np

weight_raw = open("../../weights_data/conv_0_weight.txt","r")
data_raw = open("../../input_data/input_data.txt","r")

weight_lines = weight_raw.readlines()
data_lines = data_raw.readlines()

strweight = []
strdata = []
weight = []
data = []

for lines in weight_lines:
    weight_temp = []
    strweight = lines.split()
    if (strweight[0] == "#"):
        continue
    for i in strweight:
        weight_temp.append(float(i))
    weight.append(weight_temp)

for lines in data_lines:
    data_temp = []
    strdata = lines.split()
    for i in strdata:
        data_temp.append(float(i))
    data.append(data_temp)

weight = np.array(weight)
data = np.array(data)
weight = weight.reshape(8,4,16)

out = np.zeros([84,8])
for i in range (0,84):
    for j in range (0,8):
        temp = []
        for k in range (0,4):
            temp.append(np.sum(weight[j,k,:]*data[k]))
        temp = np.array(temp)
        # print (temp)
        out[i,j] = np.sum(temp)
    data = np.delete(data, 0, axis=0)

print (out)
