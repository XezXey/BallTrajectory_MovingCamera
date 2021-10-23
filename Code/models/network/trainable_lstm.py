from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Trainable_LSTM(pt.nn.Module):
  def __init__(self, in_node, hidden, stack, 
              trainable_init, is_bidirectional, batch_size):
    super(Trainable_LSTM, self).__init__()
    # Define the model parameters
    self.in_node = in_node
    self.batch_size = batch_size
    self.hidden = hidden
    self.stack = stack+1

    self.is_bidirectional = is_bidirectional
    self.bidirectional = 2 if is_bidirectional else 1

    # Initial state
    self.trainable_init = trainable_init
    if not self.trainable_init:
      self.h, self.c = self.initial_state()
    else:
      self.h, self.c = self.initial_learnable_state()
      self.register_parameter('h', self.h)
      self.register_parameter('c', self.c)

    # Stack Recurrent Layers
    ls = [pt.nn.LSTM(input_size=in_node, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=self.is_bidirectional, dropout=0.)]

    for i in range(stack):
      ls.append(pt.nn.LSTM(input_size=hidden*self.bidirectional, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=self.is_bidirectional, dropout=0.))
    self.ls = pt.nn.Sequential(*ls)


  def forward(self, in_f, lengths, h=None, c=None):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    # Use packed sequence to speed up the RNN/LSTM
    # pack_padded_sequence => RNN => pad_packed_sequence[0] to get the data in batch
    in_f_packed = pack_padded_sequence(in_f, lengths=lengths, batch_first=True, enforce_sorted=False)
    out_packed = in_f_packed
    for idx, recurrent_block in enumerate(self.ls):
      # Pass the packed sequence to the recurrent blocks with the skip connection
      if self.trainable_init:
        init_h = self.h.repeat(1, 1, self.batch_size, 1)[idx]
        init_c = self.c.repeat(1, 1, self.batch_size, 1)[idx]
      else:
        init_h, init_c = self.initial_state()
      # print("I : ", idx, pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0].shape)
      out_packed, (h, c) = recurrent_block(out_packed, (init_h, init_c))
      # print("O : ", idx, pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0].shape)

    # Residual from recurrent block to FC
    out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-1000)[0]
    return out_unpacked, (h, c)

  def initial_state(self):
    h = Variable(pt.zeros(self.bidirectional, self.batch_size, self.hidden, dtype=pt.float32)).cuda()
    c = Variable(pt.zeros(self.bidirectional, self.batch_size, self.hidden, dtype=pt.float32)).cuda()
    return h, c

  def initial_learnable_state(self):
    # Initial the hidden/cell state as model parameters
    # 1 refer the batch size which is we need to copy the initial state to every sequence in a batch
    h = pt.nn.Parameter(pt.randn(self.stack, self.bidirectional, 1, self.hidden, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
    c = pt.nn.Parameter(pt.randn(self.stack, self.bidirectional, 1, self.hidden, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
    return h, c

