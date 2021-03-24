import numpy as np
from preprocessings import get_mfccs, load_from_pickle, save_to_pickle
from GenderClassifier import GenderClassifier
import torch
from torch.autograd import Variable

x_train_mfccs = np.asarray(get_mfccs(pickle_file="X-train-mfccs.pkl"))/100
x_test_mfccs = np.asarray(get_mfccs(pickle_file="X-test-mfccs.pkl"))/100
y_train = load_from_pickle(filename="y-train.pkl")
y_test = load_from_pickle(filename="y-test.pkl")

x_train_tensor = Variable(torch.Tensor(x_train_mfccs), requires_grad=False)
x_test_tensor = Variable(torch.Tensor(x_test_mfccs), requires_grad=False)
y_train_tensor = Variable(torch.LongTensor(y_train), requires_grad=False)
y_test_tensor = Variable(torch.LongTensor(y_test), requires_grad=False)


model = GenderClassifier()
model.load_state_dict(torch.load('model_state_dict.pkl'))
model.eval()

fc_train_in = model.get_fc_in(x_train_tensor)
fc_test_in = model.get_fc_in(x_test_tensor)

save_to_pickle(fc_train_in.detach().numpy(), "fc-train-in.pkl")
save_to_pickle(fc_test_in.detach().numpy(), "fc-test-in.pkl")