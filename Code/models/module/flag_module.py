from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from models.network.vanilla_mlp import Vanilla_MLP
from models.network.trainable_lstm import Trainable_LSTM

class Flag_Module(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size):
        super(Flag_Module, self).__init__()
        bidirectional = 2 if is_bidirectional else 1

        self.rnn = Trainable_LSTM(in_node=in_node, hidden=rnn_hidden, stack=rnn_stack, 
            trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
        self.mlp = Vanilla_MLP(in_node=bidirectional*rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
            out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)
        self.sigmoid = pt.nn.Sigmoid()
        
    def forward(self, in_f, lengths, h=None, c=None):
        out1, (h, c) = self.rnn(in_f, lengths)
        out2 = self.mlp(out1)
        out3 = self.sigmoid(out2)
        return out3, (h, c)

