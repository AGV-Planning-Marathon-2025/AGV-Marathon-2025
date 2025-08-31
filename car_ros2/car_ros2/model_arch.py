import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.seq = []  # rename fc to seq
        self.bn = nn.BatchNorm1d(input_size, affine=False)  # rename batch_norm to bn
        curr_h = input_size
        for h in hidden_size:
            self.seq.append(nn.Linear(curr_h, h))
            self.seq.append(nn.ReLU())
            curr_h = h
        self.seq.append(nn.Linear(curr_h, output_size))
        self.seq = nn.Sequential(*self.seq)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = self.bn(x)
        elif len(x.shape) == 3:
            x1 = self.bn(x.reshape((-1, x.shape[-1])))
            x = x1.reshape(x.shape)
        out = self.seq(x)
        return out
