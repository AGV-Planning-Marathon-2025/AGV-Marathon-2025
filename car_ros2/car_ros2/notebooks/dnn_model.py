import torch
import torch.nn as nn


class DNNModel(nn.Module):
    def __init__(self, dims=[39, 128, 128, 64, 1]):
        super().__init__()
        self.seq = nn.Sequential()
        self.bn = nn.BatchNorm1d(dims[0], affine=False)
        
        
        for i in range(1, len(dims)):
            self.seq.append(nn.Linear(dims[i-1], dims[i]))
            if i < len(dims) - 1: self.seq.append(nn.ReLU())


    def forward(self, x: torch.Tensor):
        if len(x.shape) == 2:
            x = self.bn(x)
        elif len(x.shape) == 3:
            x1 = self.bn(x.reshape((-1, x.shape[-1])))
            x = x1.reshape(x.shape)
        
        out = self.seq(x)
        return out