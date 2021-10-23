from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys, os
import numpy as np
import torch as pt
from torch.autograd import Variable
from models.network.vanilla_mlp import Vanilla_MLP
sys.path.append(os.path.realpath('../../'))
from utils import utils_func as utils_func

class LSTM_Agg(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_stack, mlp_hidden,
                trainable_init, is_bidirectional, batch_size):
        super(LSTM_Agg, self).__init__()
        # Define the model parameters
        self.in_node = in_node + 1
        self.out_node = out_node
        self.batch_size = batch_size
        self.rnn_hidden = rnn_hidden
        self.rnn_stack = rnn_stack + 1
        self.mlp_hidden = mlp_hidden
        self.mlp_stack = mlp_stack + 1

        self.is_bidirectional = is_bidirectional
        self.bidirectional = 2 if is_bidirectional else 1

        # Initial state
        # Shape of initial state is [N_stack, N_direction, Batch_size, N_dim]
        self.trainable_init = trainable_init
        if not self.trainable_init:
            self.hs, self.cs = self.initial_state()
        else:
            self.hs, self.cs = self.initial_learnable_state()
            self.register_parameter('h', self.hs)
            self.register_parameter('c', self.cs)

        # Stack Recurrent Layers
        ls_fw = [pt.nn.LSTMCell(input_size=self.in_node, hidden_size=rnn_hidden)]   # First layer
        ls_bw = [pt.nn.LSTMCell(input_size=self.in_node, hidden_size=rnn_hidden)]   # First layer

        for _ in range(rnn_stack):
            ls_fw.append(pt.nn.LSTMCell(input_size=rnn_hidden, hidden_size=rnn_hidden))
            ls_bw.append(pt.nn.LSTMCell(input_size=rnn_hidden, hidden_size=rnn_hidden))

        self.ls_fw = pt.nn.Sequential(*ls_fw)
        self.ls_bw = pt.nn.Sequential(*ls_bw)

        # Linear Layers
        self.mlp_fw = Vanilla_MLP(in_node=rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
            out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)

        self.mlp_bw = Vanilla_MLP(in_node=rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
            out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)

    def forward(self, in_f, lengths, mask, search_h=None):
        # Input for forward and backward
        in_f = in_f
        in_f_rev = utils_func.reverse_seq_at_lengths(seq=in_f, lengths=lengths)

        mask_h = mask[..., [0]].float()
        mask_h_rev = utils_func.reverse_seq_at_lengths(seq=mask_h, lengths=lengths)
        h_fw = pt.squeeze(search_h['first_h'], dim=-1)
        h_bw = pt.squeeze(search_h['last_h'], dim=-1)
        fw = [h_fw]
        bw = [h_bw]
        # Iterate to each time-step
        for t in range(in_f.shape[1]):
            # Rnn layer
            for idx, _ in enumerate(self.ls_fw):
                # initial h and c
                if t == 0:
                    if self.trainable_init:
                        init_hs = self.hs.repeat(1, 1, self.batch_size, 1)[idx]
                        init_cs = self.cs.repeat(1, 1, self.batch_size, 1)[idx]
                    else:
                        init_hs, init_cs = self.initial_state()

                    hs_fw, cs_fw = init_hs[0], init_cs[0]
                    hs_bw, cs_bw = init_hs[1], init_cs[1]

                if idx == 0:
                    # Forward direction
                    hs_fw, cs_fw = self.ls_fw[idx](pt.cat((in_f[:, t, :], h_fw), dim=-1), (hs_fw, cs_fw))
                    # Backward direction
                    hs_bw, cs_bw = self.ls_bw[idx](pt.cat((in_f_rev[:, t, :], h_bw), dim=-1), (hs_bw, cs_bw))
                else :
                    # Forward direction
                    hs_fw, cs_fw = self.ls_fw[idx](hs_fw, (hs_fw, cs_fw))
                    # Backward direction
                    hs_bw, cs_bw = self.ls_bw[idx](hs_bw, (hs_bw, cs_bw))

            # Accumulator
            pred_fw = self.mlp_fw(hs_fw)
            pred_bw = self.mlp_bw(hs_bw)
            h_fw = h_fw + (pred_fw * mask_h[:, t, :])
            h_bw = h_bw + (pred_bw * mask_h_rev[:, t, :])

            fw.append(h_fw)
            bw.append(h_fw)
        
        fw = pt.stack((fw), dim=1)
        bw = pt.stack((bw), dim=1)
        bw = utils_func.reverse_seq_at_lengths(seq=bw, lengths=lengths+1)

        # Residual from recurrent block to FC
        return fw, bw

    def initial_state(self):
        h = Variable(pt.zeros(self.bidirectional, self.batch_size, self.rnn_hidden, dtype=pt.float32)).cuda()
        c = Variable(pt.zeros(self.bidirectional, self.batch_size, self.rnn_hidden, dtype=pt.float32)).cuda()
        return h, c

    def initial_learnable_state(self):
        # Initial the hidden/cell state as model parameters
        # 1 refer the batch size which is we need to copy the initial state to every sequence in a batch
        h = pt.nn.Parameter(pt.randn(self.rnn_stack, self.bidirectional, 1, self.rnn_hidden, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
        c = pt.nn.Parameter(pt.randn(self.rnn_stack, self.bidirectional, 1, self.rnn_hidden, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
        return h, c


