from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt

class Vanilla_MLP(pt.nn.Module):
    def __init__(self, in_node, out_node, hidden, stack, lrelu_slope, batch_size):
        self.activation = pt.nn.LeakyReLU(lrelu_slope)

        # This will create the FC blocks by specify the input/output features
        ls = [pt.nn.Linear(in_node, hidden)]   # Original
        ls.append(self.activation)
        for i in range(stack):
            ls.append([hidden, hidden])
            ls.append(self.activation)
        ls.append(pt.nn.Linear(hidden, out_node))
        self.seq1 = pt.nn.Sequential(*ls)

    def forward(self, in_f):
        # Pass the unpacked(The hidden features from RNN) to the FC layers
        out = self.seq1(in_f)