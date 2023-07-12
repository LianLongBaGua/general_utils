import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvLSTMNet(nn.Module):
    def __init__(self):
        super(ConvLSTMNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 1 input channel, 6 output channels, 3x3 kernel
        self.fc1 = None
        self.lstm = nn.LSTM(120, 84, 1)  # 120 input features, 84 hidden size, 1 layer
        self.fc2 = nn.Linear(84, 3)  # 84 input features, 3 output features (for 3 classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # add an extra dimension for the single channel
        #x = F.relu(self.conv1(x))

        x = x.view(x.size(0), -1)  # flatten the tensor
        #print(f'Shape of x before fc1: {x.shape}')
        if self.fc1 is None or self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 120).to(x.device)
        x = F.relu(self.fc1(x))
        x, _ = self.lstm(x.view(1, x.size(0), -1))  # LSTM expects a 3D input (seq_len, batch, input_size)
        x = self.fc2(x.view(x.size(1), -1))
        return x