# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 09:34:30 2020

@author: wuwen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

class ActorCritic(torch.nn.Module):
    
    def __init__(self, inSize, outSize, layers):
        super(ActorCritic, self).__init__()
        self.layersA = torch.nn.ModuleList([])
        self.layersC = torch.nn.ModuleList([])
        for x in layers:
            self.layersA.append(torch.nn.Linear(inSize, x)) ## (4, 200)
            self.layersC.append(torch.nn.Linear(inSize, x))
            inSize = x
        self.layersA.append(nn.Linear(inSize, outSize)) ## (200, 2)
        self.layersC.append(nn.Linear(inSize, 1))
        self.sm = torch.nn.Softmax(0)
        
    def forward(self, x):
        xA = self.layersA[0](x) ## (4, ) x (4, 200) -> (200, )
        xC = self.layersC[0](x)
        for i in range(1, len(self.layersA)):
            xA = torch.tanh(xA)
            xA = self.layersA[i](xA) ## (200, ) x (200, 2) -> (2, ) G ou D
            xC = F.relu( self.layersC[i](xC) )
        return xC, self.sm(xA)