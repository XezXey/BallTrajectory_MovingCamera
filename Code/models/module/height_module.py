from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from models.network.vanilla_mlp import Vanilla_MLP
from models.network.self_attention import Self_Attention
from models.network.trainable_lstm import Trainable_LSTM
from models.network.lstm_agg import LSTM_Agg

class Height_Module(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size, attn=False):
        super(Height_Module, self).__init__()
        bidirectional = 2 if is_bidirectional else 1

        self.rnn_layer = Trainable_LSTM(in_node=in_node, hidden=rnn_hidden, stack=rnn_stack, 
            trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
        self.attn = attn
        if self.attn:
            self.attention = Self_Attention()
        self.mlp = Vanilla_MLP(in_node=bidirectional*rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
            out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)
        
    def forward(self, in_f, lengths, h=None, c=None, search_h=None, mask=None):
        out1, (h, c) = self.rnn_layer(in_f, lengths, search_h)
        # Time-attention
        if self.attn:
            out1 = self.attention(out1)
        out2 = self.mlp(out1)
        return out2, (h, c)

class Height_Module_Agg(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size, attn=False):
        super(Height_Module_Agg, self).__init__()
        bidirectional = 2 if is_bidirectional else 1

        self.rnn_layer = LSTM_Agg(in_node=in_node, out_node=out_node, 
            rnn_hidden=rnn_hidden, rnn_stack=rnn_stack, mlp_stack=mlp_stack, mlp_hidden=mlp_hidden, 
            trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
        self.attn = attn
        if self.attn:
            self.attention = Self_Attention()

        self.rnn_layer2 = Trainable_LSTM(in_node=1, hidden=rnn_hidden, stack=rnn_stack, 
            trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
        self.mlp = Vanilla_MLP(in_node=bidirectional * rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
            out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)
        
    def forward(self, in_f, lengths, h=None, c=None, search_h=None, mask=None):
        fw, bw = self.rnn_layer(in_f, lengths, search_h=search_h, mask=mask)
        return fw, bw

class Height_Module_(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size, attn=False):
        super(Height_Module_Agg, self).__init__()
        bidirectional = 2 if is_bidirectional else 1

        self.rnn_layer = LSTM_Agg(in_node=in_node, out_node=out_node, 
            rnn_hidden=rnn_hidden, rnn_stack=rnn_stack, mlp_stack=mlp_stack, mlp_hidden=mlp_hidden, 
            trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
        self.attn = attn
        if self.attn:
            self.attention = Self_Attention()

        self.rnn_layer2 = Trainable_LSTM(in_node=1, hidden=rnn_hidden, stack=rnn_stack, 
            trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
        self.mlp = Vanilla_MLP(in_node=bidirectional * rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
            out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)
        
    def forward(self, in_f, lengths, h=None, c=None, search_h=None, mask=None):
        fw, bw = self.rnn_layer(in_f, lengths, search_h=search_h, mask=mask)
        fw, (_, _) = self.rnn_layer2(in_f=fw, lengths=lengths)
        fw = self.mlp(in_f=fw)
        return fw, bw