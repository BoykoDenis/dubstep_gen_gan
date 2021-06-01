import torch
import torch.nn as nn

#class testrnn(nn.Module):

#    def __init__():

#        self.rnn1 = nn.RNN()

rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)

print(input)
print(input.shape)
print(h0)
print(h0.shape)

print(output)
print(output.shape)