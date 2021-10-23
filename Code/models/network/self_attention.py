from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt

class Self_Attention(pt.nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()

        # This will create a self-attention block following the CVPR_PFNet
        self.softmax = pt.nn.Softmax(dim=-1)
        self.gamma = pt.nn.Parameter(pt.ones(1))

    def forward(self, in_f):
        '''
        Input : 
            1. in_f : features from BiLSTM (batch_size, seq_len, f_dim)
        Output :
            1. out : time attentive features
        '''
        batch_size, seq_len, f_dim = in_f.size()
        proj_query = in_f.view(batch_size, seq_len, f_dim)  # B X L X C
        proj_key = in_f.view(batch_size, seq_len, f_dim).permute(0, 2, 1)   # B X C X L
        attn_energy = pt.bmm(proj_query, proj_key)  # B X L X L
        attn_w = self.softmax(attn_energy)
        
        proj_value = in_f.view(batch_size, seq_len, f_dim)  # B X L X C
        out = pt.bmm(attn_w, proj_value)

        out = self.gamma * out + in_f
        return out
