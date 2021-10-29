from __future__ import print_function
import warnings

from numpy.lib.recfunctions import require_fields
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt

class Optimization(pt.nn.Module):
    def __init__(self, shape, name):
        super(Optimization, self).__init__()
        
        self.params = pt.nn.Parameter(data=pt.rand(size=shape).cuda(), requires_grad=True)
        self.optimizer = pt.optim.Adam(self.parameters(), lr=1e-1)
        self.lr_scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10)
        self.lrelu = pt.nn.LeakyReLU()
        self.name = name
        
    def forward(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        #self.info()
        self.optimizer.step()
        self.lr_scheduler.step(loss)
        #for param_group in self.optimizer.param_groups:
        #    print("LR : ", param_group['lr'], "Loss : ", loss)

    def get_params(self):
        if self.name != 'init_first_h' and self.name != 'init_last_h':
            # Constraint/Normalize the latent
            #print(self.params)
            params_constrainted = self.params / (pt.sqrt(pt.sum(self.params**2, dim=-1, keepdims=True)) + 1e-16)
            #print(params_constrainted)
            return params_constrainted
        elif self.name == 'init_first_h' or self.name == 'init_last_h':
            params_constrainted = self.params
            return params_constrainted
        else:
            return self.params

    def set_params(self, params):
        if list(self.params.shape) != list(params.shape):
            raise Exception("Different params's shape assignment")
        else:
            self.params = pt.nn.Parameter(data=params, requires_grad=True)

    def get_name(self):
        return self.name

    def info(self):
        print("Name : ", self.name)
        print("Grads : ", self.params.grad)
        print("Params : ", self.params)