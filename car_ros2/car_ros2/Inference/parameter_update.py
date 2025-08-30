import torch
import numpy as np

def grad_optim(model, X, grad_rate, param_indices, bounds):
    X = torch.autograd.Variable(X, requires_grad=True)
    model.zero_grad()
    preds = model(X)
    grads = torch.autograd.grad(preds[0,0], X, retain_graph=True)[0].data
    X = X.detach()
    for i in param_indices:
        low, high = bounds[i]
        X[0,i] = max(low, min(high, X[0,i].item() + grad_rate * grads[0,i].item()))
    return X, grads

def regret_entry(new_V, prev_V, param_dict):
    return [new_V - prev_V, new_V, prev_V] + [float(param_dict[k]) for k in sorted(param_dict)]