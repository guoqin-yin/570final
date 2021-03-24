import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import pickle
from GenderClassifier import Args, GenderClassifier
from preprocessings import get_datases, get_mfccs, save_to_pickle, load_from_pickle, regularize_data
from constants import CLASSES

import matplotlib.pyplot as plt

feature_loading = 'load_from_pkl' # option: load_from_flac or load_from_pkl
feature_store = True

model_loading = 'load_from_pkl' # option: load_from_pkl or train


if feature_loading == 'load_from_flac':
    (x_train, y_train), (x_test, y_test) = get_datases(class_type='speaker')

    x_train_mfccs = get_mfccs(x_train)
    x_test_mfccs = get_mfccs(x_test)

    if feature_store:
        save_to_pickle(x_train_mfccs, "X-train-mfccs.pkl")
        save_to_pickle(x_test_mfccs, "X-test-mfccs.pkl")
        save_to_pickle(y_train, "y-train.pkl")
        save_to_pickle(y_test, "y-test.pkl")

else:
    x_train_mfccs = get_mfccs(pickle_file="X-train-mfccs.pkl")
    x_test_mfccs = get_mfccs(pickle_file="X-test-mfccs.pkl")
    y_train = load_from_pickle(filename="y-train.pkl")
    y_test = load_from_pickle(filename="y-test.pkl")


x_train_mfccs = regularize_data(x_train_mfccs)
x_test_mfccs = regularize_data(x_test_mfccs)

x_train_tensor = Variable(torch.Tensor(x_train_mfccs), requires_grad=False)
x_test_tensor = Variable(torch.Tensor(x_test_mfccs), requires_grad=False)
y_train_tensor = Variable(torch.LongTensor(y_train), requires_grad=False)
y_test_tensor = Variable(torch.LongTensor(y_test), requires_grad=False)

lstm_model = GenderClassifier(len(np.bincount(y_train)))
# optimizer = torch.optim.SGD(lstm_model.parameters(), lr=1e-4, weight_decay=1e-6, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3, weight_decay=1e-3)
loss_function = nn.CrossEntropyLoss()


def get_batch_data(batch_size):
    rand_index = np.random.randint(0, len(x_train_mfccs), batch_size).tolist()
    # print(type(rand_index.tolist()))
    x_train_batch = [x_train_mfccs[i][:] for i in rand_index]
    y_train_batch = [y_train[i] for i in rand_index]
    x_train_tensor = Variable(torch.Tensor(x_train_batch), requires_grad=False)
    y_train_tensor = Variable(torch.LongTensor(y_train_batch), requires_grad=False)
    return x_train_tensor, y_train_tensor


# get_batch_data(16)
# x_train, y_train = get_batch_data(16)
# print(x_train.size())

loss_data  = []
accuracy_data = []

for cur_iter in range(200):
    print("iter: {:2d}".format(cur_iter), end=", ")
    # lstm_model.zero_grad()
    # x_batch, y_batch = get_batch_data(256)
    outputs = lstm_model(x_train_tensor)
    optimizer.zero_grad()
    # print(outputs.size())
    loss = loss_function(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    _, outputs_label = outputs.max(dim=1)

    accuracy = int(sum(outputs_label == y_train_tensor))/len(y_train_tensor)
    print("accuray: {:.2f}, loss: {:.2e}".format(accuracy, loss))
    # print("output: {}, label: {}".format(outputs_label[0], y_batch[0]))

    loss_data.append(loss.values)
    accuracy_data.append(accuracy)

outputs = lstm_model(x_test_tensor)
_, y_pred = outputs.max(dim=1)
accuracy = int(sum(y_pred == y_test_tensor))/len(y_test_tensor)
print("@@@ Test data accuracy: {}".format(accuracy))


# print("Model state_dict:")
# print(lstm_model.state_dict())

torch.save(lstm_model.state_dict(), 'model_state_dict.pkl')
print("# CLASSES: {}".format(len(np.bincount(y_train))))

# model = lstm_model()
# model.load_state_dict(torch.load('model_state_dict.pkl'))
# model.eval()

# plt.plot(accuracy)
# plt.show()
