from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt

class LSTM_TrainableInit(pt.nn.Module):
  def __init__(self, in_node, out_node, trainable_init,
              rnn_hidden, rnn_stack, bidirectional,
              model, batch_size):
    super(LSTM_TrainableInit, self).__init__()
    # Define the model parameters
    self.in_node = in_node
    self.out_node = out_node
    self.batch_size = batch_size
    self.rnn_hidden = rnn_hidden
    self.rnn_stack = rnn_stack
    self.model = model

    self.is_bidirectional = bidirectional
    if bidirectional:
      self.bidirectional = 2
    else:
      self.bidirectional = 1

    # For a  initial state
    self.trainable_init = trainable_init
    if not self.trainable_init:
      self.h, self.c = self.initial_state()
    else:
      self.h, self.c = self.initial_learnable_state()
      self.register_parameter('h', self.h)
      self.register_parameter('c', self.c)

    # This will create the Recurrent blocks by specify the input/output features
    self.recurrent_stacked = [self.input_size] + [self.hidden_dim] * self.n_stack
    # Define the layers
    # Recurrent layer with Bi-directional : in_node*2 for takes 2 directional from previous layers
    self.recurrent_blocks = pt.nn.Sequential([self.create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=True) if in_f == self.input_size
                                              else self.create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=False)
                                              for in_f, hidden_f in zip(self.recurrent_stacked, self.recurrent_stacked[1:])])

  def forward(self, in_f, lengths, hidden=None, cell_state=None):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    # Use packed sequence to speed up the RNN/LSTM
    # pack_padded_sequence => RNN => pad_packed_sequence[0] to get the data in batch
    in_f_packed = pack_padded_sequence(in_f, lengths=lengths, batch_first=True, enforce_sorted=False)
    out_packed = in_f_packed
    for idx, recurrent_block in enumerate(self.recurrent_blocks):
      # print("IDX = {}".format(idx), self.h[idx], self.c[idx])
      # Pass the packed sequence to the recurrent blocks with the skip connection
      if self.trainable_init:
        init_h = self.h.repeat(1, 1, self.batch_size, 1)[idx]
        init_c = self.c.repeat(1, 1, self.batch_size, 1)[idx]
      else:
        init_h, init_c = self.initial_state()
      # Only first time that no skip connection from input to other networks
      # print("I : ", idx, pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0].shape)
      out_packed, (hidden, cell_state) = recurrent_block(out_packed, (init_h, init_c))
      # print("O : ", idx, pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0].shape)

    # Residual from recurrent block to FC
    out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0]
    return out, (hidden, cell_state)

  def create_fc_block(self, in_f, out_f, is_last_layer=False):
    # Auto create the FC blocks
    if is_last_layer:
      return pt.nn.Sequential(
        pt.nn.Linear(in_f, out_f, bias=True),)
    else :
      return pt.nn.Sequential(
        pt.nn.Linear(in_f, out_f, bias=True),
        pt.nn.LeakyReLU(negative_slope=0.01),
      )

  def create_recurrent_block(self, in_f, hidden_f, n_stack):
    # Stack Recurrent Layers
    for i in range(n_stack):
      return pt.nn.LSTM(input_size=in_f, hidden_size=hidden_f, num_layers=1, batch_first=True, bidirectional=self.is_bidirectional, dropout=0.)

    pt.nn.LSTM(input_size=in_f*self.bidirectional, hidden_size=hidden_f, num_layers=1, batch_first=True, bidirectional=self.bidirectional_flag, dropout=0.)

  def initial_state(self):
    h = Variable(pt.zeros(self.n_layers*self.bidirectional, self.batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    c = Variable(pt.zeros(self.n_layers*self.bidirectional, self.batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return h, c

  def initial_learnable_state(self):
    # Initial the hidden/cell state as model parameters
    # 1 refer the batch size which is we need to copy the initial state to every sequence in a batch
    h = pt.nn.Parameter(pt.randn(self.n_stack, self.n_layers*self.bidirectional, 1, self.hidden_dim, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
    c = pt.nn.Parameter(pt.randn(self.n_stack, self.n_layers*self.bidirectional, 1, self.hidden_dim, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
    return h, c

