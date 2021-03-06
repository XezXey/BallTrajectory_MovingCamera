from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from models.network.vanilla_mlp import Vanilla_MLP
from models.network.self_attention import Self_Attention
from models.network.trainable_lstm import Trainable_LSTM

class Refinement_Module(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size, attn=False):
        super(Refinement_Module, self).__init__()
        bidirectional = 2 if is_bidirectional else 1

        self.rnn = Trainable_LSTM(in_node=in_node, hidden=rnn_hidden, stack=rnn_stack, 
            trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
        self.attn = attn
        if self.attn:
            self.attention = Self_Attention()
        self.mlp = Vanilla_MLP(in_node=bidirectional*rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
            out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)
        
    def forward(self, in_f, lengths, h=None, c=None):
        out1, (h, c) = self.rnn(in_f, lengths)
        # Time-attention
        if self.attn:
            out1 = self.attention(out1)
        out2 = self.mlp(out1)
        return out2, (h, c)

class Stack_Refinement_Module(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size, n_refinement, attn=False):
        super(Stack_Refinement_Module, self).__init__()
        bidirectional = 2 if is_bidirectional else 1

        self.refinement = []
        for _ in range(n_refinement):
            rnn = Trainable_LSTM(in_node=in_node, hidden=rnn_hidden, stack=rnn_stack, 
                trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
            mlp = Vanilla_MLP(in_node=bidirectional*rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
                out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)
            refinement_ = pt.nn.ModuleList([rnn, mlp])
            self.refinement.append(refinement_)
            self.refinement = pt.nn.ModuleList(self.refinement)

    def forward(self, in_f, lengths, h=None, c=None):
        for net in self.refinement:
            out1, _ = net[0](in_f, lengths)
            out2 = net[1](out1)
            in_f[..., [0]] = in_f[..., [0]] + out2

        return out2, (h, c)