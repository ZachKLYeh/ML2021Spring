import numpy as np
from math import pi
from collections import defaultdict
from autograd_lib import autograd_lib

import torch
import torch.nn as nn
import model

#key
key = 0

# load checkpoint and data corresponding to the key
model = model.MathRegressor()
autograd_lib.register(model)

data = torch.load('data.pth')[key]
model.load_state_dict(data['model'])
train, target = data['data']

#set criterion
criterion = nn.MSELoss()

# reset compute dictionaries
activations = defaultdict(int)
hess = defaultdict(float)

#compute gradient norm
model.train()
model.zero_grad()
output = model(train)
loss = criterion(output, target)
loss.backward()

grads = []
for p in model.regressor.children():
    if isinstance(p, nn.Linear):
        param_norm = p.weight.grad.norm(2).item()
        grads.append(param_norm)

grad_mean = np.mean(grads)

# helper function to save activations
def save_activations(layer, A, _):
    '''
    A is the input of the layer, we use batch size of 6 here
    layer 1: A has size of (6, 1)
    layer 2: A has size of (6, 128)
    '''
    activations[layer] = A

# helper function to compute Hessian matrix
def compute_hess(layer, _, B):
    '''
    B is the backprop value of the layer
    layer 1: B has size of (6, 128)
    layer 2: B ahs size of (6, 1)
    '''
    A = activations[layer]
    BA = torch.einsum('nl,ni->nli', B, A) # do batch-wise outer product

    # full Hessian
    hess[layer] += torch.einsum('nli,nkj->likj', BA, BA) # do batch-wise outer product, then sum over the batch


#compute minimum ratio
model.zero_grad()

with autograd_lib.module_hook(save_activations):
    output = model(train)
    loss = criterion(output, target)
    
with autograd_lib.module_hook(compute_hess):
    autograd_lib.backward_hessian(output, loss='LeastSquares')

layer_hess = list(hess.values())
minimum_ratio = []

# compute eigenvalues of the Hessian matrix
for h in layer_hess:
    size = h.shape[0] * h.shape[1]
    h = h.reshape(size, size)
    h_eig = torch.symeig(h).eigenvalues # torch.symeig() returns eigenvalues and eigenvectors of a real symmetric matrix
    num_greater = torch.sum(h_eig > 0).item()
    minimum_ratio.append(num_greater / len(h_eig))

ratio_mean = np.mean(minimum_ratio) # compute mean of minimum ratio

print(grad_mean, ratio_mean)