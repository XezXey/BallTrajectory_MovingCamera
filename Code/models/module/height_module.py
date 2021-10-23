from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from utils import utils_func as utils_func
from models.network.vanilla_mlp import Vanilla_MLP
from models.network.self_attention import Self_Attention
from models.network.trainable_lstm import Trainable_LSTM
from models.network.lstm_agg import LSTM_Agg

class Height_Module(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size, args, attn=False):
        super(Height_Module, self).__init__()

        # Property
        self.i_s = args.pipeline['height']['i_s']
        self.o_s = args.pipeline['height']['o_s']
        self.args = args

        # Network
        bidirectional = 2 if is_bidirectional else 1
        self.rnn = Trainable_LSTM(in_node=in_node, hidden=rnn_hidden, stack=rnn_stack, 
            trainable_init=trainable_init, is_bidirectional=is_bidirectional, batch_size=batch_size)
        self.attn = attn
        if self.attn:
            self.attention = Self_Attention()
        self.mlp = Vanilla_MLP(in_node=bidirectional*rnn_hidden, hidden=mlp_hidden, stack=mlp_stack, 
            out_node=out_node, batch_size=batch_size, lrelu_slope=0.01)

        
    def forward(self, in_f, lengths, h=None, c=None, search_h=None, mask=None):
        out1, (h, c) = self.rnn(in_f, lengths, search_h)
        # Time-attention
        if self.attn:
            out1 = self.attention(out1)
        out2 = self.mlp(out1)
        
        height = output_space(pred_h=out2, lengths=lengths+1 if ((self.i_s == 'dt' or self.i_s == 'dt_intr' or self.i_s == 'dt_all') and self.o_s == 'dt') else lengths, module='height', args=self.args)

        return height

class Height_Module_Agg(pt.nn.Module):
    def __init__(self, in_node, out_node, rnn_hidden, rnn_stack, mlp_hidden, mlp_stack, 
                trainable_init, is_bidirectional, batch_size, args, attn=False):
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
        h_fw, h_bw = self.rnn_layer(in_f, lengths, search_h=search_h, mask=mask)
        w_ramp = utils_func.construct_w_ramp(weight_template=pt.zeros(size=(in_f.shape[0], in_f.shape[1]+1, 1)), lengths=lengths+1)
        height = pt.sum(pt.cat((h_fw, h_bw), dim=2) * w_ramp, dim=2, keepdims=True)
        return height

def output_space(pred_h, lengths, module, args, search_h=None):
  '''
  Aggregate the height-prediction into (t, dt)
  Input : 
    1. height : height in shape (batch, seq_len, 1)
    2. lengths : lengths of each seq to determine the wegiht size, and position to reverse the seq(always comes in t-space)
    3. search_h(optional) : In scenario which is trajectory didn't start/end on the ground. (Handling by search for h)
      in shape {'first_h' : (batch, 1, 1), 'last_h' : (batch, 1, 1)}
  Output : 
    1. height :height after aggregation into t-space
  '''
  i_s = args.pipeline[module]['i_s']
  o_s = args.pipeline[module]['o_s']

  if o_s == 't':
    height = pred_h
  elif o_s == 'dt' or 't_dt':
    if ((i_s == 'dt') or (i_s == 'dt_intr') or (i_s == 'dt_all')):
      dh = pred_h
    elif i_s == 't_dt' or i_s == 't':
      dh = pred_h[:, :-1, :]

    if o_s == 't_dt' and i_s == 't_dt':
      dh = pred_h[:, :-1, [1]]

    # Aggregate the dt output with ramp_weight
    w_ramp = utils_func.construct_w_ramp(weight_template=pt.zeros(size=(dh.shape[0], dh.shape[1]+1, 1)), lengths=lengths)

    if search_h is None:
      first_h = pt.zeros(size=(dh.shape[0], 1, 1)).cuda()
      last_h = pt.zeros(size=(dh.shape[0], 1, 1)).cuda()
    else:
      first_h = search_h['first_h']
      last_h = search_h['last_h']

    # forward aggregate
    h_fw = utils_func.cumsum(seq=dh, t_0=first_h)
    # backward aggregate
    pred_h_bw = utils_func.reverse_seq_at_lengths(seq=-dh, lengths=lengths-1) # This fn required len(seq) of dt-space
    h_bw = utils_func.cumsum(seq=pred_h_bw, t_0=last_h)
    h_bw = utils_func.reverse_seq_at_lengths(seq=h_bw, lengths=lengths) # This fn required len(seq) of t-space(after cumsum)
    height = pt.sum(pt.cat((h_fw, h_bw), dim=2) * w_ramp, dim=2, keepdims=True)

    if o_s == 't_dt':
      height = (pred_h[..., [0]] * 0.5) + (height * 0.5)

  # Hard constraint on Height (y > 0)
  if args.pipeline['height']['constraint_y'] == 'relu':
    relu = pt.nn.ReLU()
    height = relu(height)
  elif args.pipeline['height']['constraint_y'] == 'softplus':
    softplus = pt.nn.Softplus()
    height = softplus(height)
  elif args.pipeline['height']['constraint_y'] == 'lrelu':
    softplus = pt.nn.LeakyReLU(negative_slope=0.01)
    height = softplus(height)
  else:
    pass

  return height

