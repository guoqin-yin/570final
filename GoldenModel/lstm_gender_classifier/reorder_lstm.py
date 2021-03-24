import shutil

lstm_data = {
    "weight_hh" : "weights_data/lstm_weight_hh_l0.txt",
    "weight_ih" : "weights_data/lstm_weight_ih_l0.txt"
    # "bias_hh"   : "weights_data/lstm_bias_hh_l0.txt",
    # "bias_ih"   : "weights_data/lstm_bias_ih_l0.txt"
}

HIDDEN_NODES = 16
FILE_POS = 12

path = lstm_data["weight_ih"][:FILE_POS+1] + 'temp_' + lstm_data["weight_ih"][FILE_POS+1:]
wfile = open(path, "w+")
with open(lstm_data["weight_ih"], "r") as f:
    wfile.write(f.readline())
    while True:
        data = f.readline()
        if not data:
            break
        data = data[:-1].split(" ")
        data += ["0" for _ in range(HIDDEN_NODES-len(data))]
        data[-1] += "\n"
        wfile.write(" ".join(data))
wfile.close()

file1 = lstm_data["weight_ih"]
file2 = lstm_data["weight_ih"][:FILE_POS+1] + 'backup_' + lstm_data["weight_ih"][FILE_POS+1:]
file3 = lstm_data["weight_ih"][:FILE_POS+1] + 'temp_' + lstm_data["weight_ih"][FILE_POS+1:]

shutil.move(file1, file2)
shutil.move(file3, file1)

keys = lstm_data.keys()
for key in lstm_data.keys():
    path = lstm_data[key][:FILE_POS+1] + 'r_' + lstm_data[key][FILE_POS+1:]
    wfile = open(path, "w+")
    with open(lstm_data[key], "r") as f:
        wfile.write(f.readline())
        for k in range(4):
            data = []
            for i in range(HIDDEN_NODES):
                data = data + f.readline()[:-1].split(' ')
                # print(data)
            wfile.write(" ".join(data[::-1]))
            wfile.write("\n")
    wfile.close()
