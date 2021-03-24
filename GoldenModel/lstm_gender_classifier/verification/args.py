class Args(object):
    def __init__(self):
        self.NUM_MFCC   = 16
        self.NUM_FRAME  = 40
        self.NUM_DATA   = 271
        # conv layer params
        self.conv1_out_channels = 16
        self.conv1_kernel_size  = 4

        self.conv2_out_channels = 8
        self.conv2_kernel_size = 2

        # Maxpool layer
        self.MP_kernel_size     = 8

        # LSTM layer params
        self.num_memory_cts         = 16
        self.lstm_input_size        = 8
        self.lstm_sequence_length   = 10
        self.lstm_num_layers        = 1
        self.lsrm_num_classes       = 10

        # fc layer
        self.fc1_in_size        = 160
        self.fc1_out_size       = 64
        self.fc2_in_size        = 64
        self.fc2_out_size       = 32