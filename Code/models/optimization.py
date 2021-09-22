from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt

class Optimization(pt.nn.Module):
    def __init__(self, shape, name):
        super(Optimization, self).__init__()
        
        self.params = pt.nn.Parameter(data=pt.randn(size=shape).cuda(), requires_grad=True)
        self.params_ = pt.nn.ParameterList()
        self.params_.append(self.params)
        self.optimizer = pt.optim.Adam(self.parameters(), lr=0.1)
        self.lr_scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10)
        self.name = name
        
    def forward(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.lr_scheduler.step(loss)
        #for param_group in self.optimizer.param_groups:
        #    print("LR : ", param_group['lr'], "Loss : ", loss)

    def get_params(self):
        if self.name != 'init_h':
            # Constraint/Normalize the latent
            #print(self.params)
            params_constrainted = self.params / (pt.sqrt(pt.sum(self.params**2, dim=-1, keepdims=True)) + 1e-16)
            #print(params_constrainted)
            return params_constrainted

        else:
            return self.params

    def get_name(self):
        return self.name
