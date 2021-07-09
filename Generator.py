import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers)

    def forward(self, x):

        x = self.rnn1(x)

        return x


