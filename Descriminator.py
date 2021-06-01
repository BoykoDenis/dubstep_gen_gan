import torch
import torch.nn as nn


class Descriminator(nn.Module):

    def __init__(self, input_fetures, memory_factor):

        self.rnn1 = nn.RNN(input_fetures, memory_factor, 3)

