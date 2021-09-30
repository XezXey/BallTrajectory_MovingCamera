from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from models.vanilla_mlp import Vanilla_MLP
from models.self_attention import Self_Attention
from models.trainable_lstm import Trainable_LSTM

class Refinement_Module(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size, attn=False):
        super(Refinement_Module, self).__init__()
        self.res_block

        
    def forward(self, in_f, lengths, h=None, c=None):
        return out2, (h, c)

class Residual_Block(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size, attn=False):

        bidirectional = 2 if is_bidirectional else 1
        self.rnn = Trainable_LSTM(in_node=in_node, hidden=rnn_hidden, stack=rnn_stack, 
            trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
        self.attention = Self_Attention()
        self.attn = attn
        self.mlp = Vanilla_MLP(in_node=bidirectional*rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
            out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)

    def forward(self, in_f, lengths, h=None, c=None):
        print(in_f.shape)
        out1, (h, c) = self.rnn(in_f, lengths)
        # Time-attention
        if self.attn:
            out1 = self.attention(out1)
        out2 = self.mlp(out1)
        print(out2.shape)
        input()
        