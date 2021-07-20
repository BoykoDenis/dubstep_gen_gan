import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers)

    def forward(self, x):

        x, h_out = self.rnn1(x)

        return x, h_out

    def running_forward(self, x, h_out):

        x, h_out = self.rnn1(x, h_out)
        return x, h_out


