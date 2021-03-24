import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from constants import NUM_MFCC, MAX_CLASSES

class Args(object):
    def __init__(self):
        # environment set
        self.is_cuda = False
        self.seed = 0

        # conv layer params
        self.conv1_out_channels = 8
        # self.conv2_out_channels = 8

        # LSTM layer params
        self.num_memory_cts = 16
        # self.num_memory_cts = 32
        self.input_size = 5
        self.sequence_length = 5
        self.batch_size = 1
        self.num_layers = 1

        # fc layer
        self.fc_in_size = 592  # equals to 41*self.num_memory_cts
        self.fc1_out_size = 128
        self.fc2_out_size = 32
        # self.fc_in_size = 1000
        # self.fc1_out_size = 500
        # self.fc2_out_size = 250

        self.fft1_block_size = 2
        self.fft2_block_size = 10


args = Args()
args.is_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)


class FFT_Linear(nn.Module):
    def __init__(self, in_channels, out_channels, block_size):
        super(FFT_Linear, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_size = block_size
        self.cirmat_size = [int(self.out_channels/self.block_size), int(self.in_channels/self.block_size), block_size]
        print(self.cirmat_size)

        self.cir_weights = nn.Parameter(torch.rand(self.cirmat_size))
    
    def forward(self, x):
        # print("input size: ", x.size())
        block_size = self.cirmat_size[2]
        x_input = [x[:,k*block_size:(k+1)*block_size] for k in range(self.cirmat_size[1])] 
        # print(x_input[0].size())
        a = [torch.zeros((x.size(0), block_size)) for _ in range(self.cirmat_size[0])]

        for i in range(self.cirmat_size[0]):
            for j in range(self.cirmat_size[1]):
                weights = self.cir_weights[i, j]
                weights = torch.stack([weights, torch.zeros(block_size)], dim=0)
                a_fft_temp = torch.fft(torch.Tensor(weights.T), signal_ndim=1)
                a_fft = torch.zeros((x.size(0), a_fft_temp.size(0), a_fft_temp.size(1)))
                for s in range(a_fft.size(0)):
                    a_fft[s,:,:] = a_fft_temp

                # print(a_fft.size())
                x_in = x_input[j]
                # print(x_in.size())
                x_in = torch.stack([x_in, torch.zeros((x.size(0), block_size))], dim=2)
                # print(x_in.size())
                b_fft = torch.fft(torch.Tensor(x_in), signal_ndim=1)


                ab_r_1 = a_fft[:,:,0] * b_fft[:,:,0]
                ab_r_2 = a_fft[:,:,1] * b_fft[:,:,1]
                ab = torch.add(a_fft[:,:,0], a_fft[:,:,1]) *  torch.add(b_fft[:,:,0], b_fft[:,:,1])

                res_r = ab_r_1 - ab_r_2
                res_i = ab - ab_r_1 - ab_r_2
                # print(res_i.size())
                # print(res_r.size())
                res = torch.stack([res_r, res_i], axis=2)
                # print(res.size())
                iff_res = torch.ifft(res, signal_ndim=1)

                a[i] += iff_res[:,:,0]

        output = a[0]
        for tensor in a[1:]:
            output = torch.cat((output, tensor), dim=1)
            # print(output.size())
        # print("fft_fc done, output size: ", output.size())
        # print(self.cir_weights[0,0])
        return output


class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.args = args

        # input shape = (batch_size, input_channels, sequence_length)
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=NUM_MFCC, out_channels=self.args.conv1_out_channels, kernel_size=4, bias=False),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=self.args.conv1_out_channels, out_channels=self.args.conv2_out_channels, kernel_size=2, bias=False),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=16)
            
        )

        # self.lstm_prev = nn.LSTM(input_size=NUM_MFCC, hidden_size=self.args.conv2_out_channels, batch_first=True)
        # self.pool = nn.MaxPool1d(kernel_size=16)
        # self.lstm = nn.LSTM(input_size=self.args.conv2_out_channels, hidden_size=args.num_memory_cts, batch_first=True)
        self.lstm = nn.LSTM(input_size=self.args.conv1_out_channels, hidden_size=self.args.num_memory_cts, batch_first=True, bias=False)
        # self.conv_after_lstm = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=4)
        
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=self.args.fc_in_size, out_features=MAX_CLASSES, bias=False),
            # nn.Linear(in_features=self.args.fc2_out_size, out_features=self.args.num_classes),
            # nn.Linear(in_features=self.args.fc_in_size, out_features=self.args.fc1_out_size),
            # nn.Linear(in_features=self.args.fc1_out_size, out_features=self.args.fc2_out_size),
            # nn.Linear(in_features=self.args.fc2_out_size, out_features=self.args.num_classes),
            nn.Softmax(dim=1)
        )

        # self.fft_fc = nn.Sequential(
        #     nn.Flatten(start_dim=1, end_dim=-1),
        #     FFT_Linear(in_channels=self.args.fc_in_size, out_channels=self.args.num_classes, block_size=2),
        #     # FFT_Linear(in_channels=self.args.fc2_out_size, out_channels=self.args.num_classes, block_size=2),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x):

        # hprev = torch.zeros((1, x.size(0), 16))
        # cprev = torch.zeros((1, x.size(0), 16))

        h0 = torch.zeros((1, x.size(0), self.args.num_memory_cts))
        c0 = torch.zeros((1, x.size(0), self.args.num_memory_cts))

        cout = self.conv(x)
        lstm_out, _ = self.lstm(cout.transpose(1, 2), (h0, c0))

        # x = x.transpose(1, 2)
        # cout, _ = self.lstm_prev(x, (hprev, cprev))
        # lstm_out, _ = self.lstm(cout, (h0, c0))
        # lstm_out = self.pool(lstm_out)

        # ====================
        # last_conv_in = lstm_out.transpose(1, 2)
        # lstm_out = self.conv_after_lstm(last_conv_in)
        # ====================
        fc_out = self.fc(lstm_out)

        return fc_out

    def get_conv_out(self, x):
        cout = self.conv(x)
        return cout

    def get_lstm_out(self, x):
        h0 = torch.zeros((1, x.size(0), self.args.num_memory_cts))
        c0 = torch.zeros((1, x.size(0), self.args.num_memory_cts))
        cout = self.conv(x)
        lstm_out, _ = self.lstm(cout.transpose(1, 2), (h0, c0))
        return poolout, lstm_out
