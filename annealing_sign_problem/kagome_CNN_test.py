import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import numpy as np

layout = torch.tensor([[-1,0,2,11],[3,5,9,-1],[1,4,6,-1],[7,-1,8,10]])
input_spins = torch.tensor([-1,1,1,1,-1,1,-1,-1,-1,-1,1,1,0])
print(layout)
print(input_spins)

#dim1 = layout.size()[0]
#dim2 = layout.size()[1]

#layout = layout.view(dim1*dim2)
#print(layout)

#layout = input_spins[layout]
#print(layout)

#input_spins = layout.view(dim1, dim2)

#print(input_spins)

dim1 = layout.size()[0]
dim2 = layout.size()[1]

layout = layout.view(dim1*dim2)
input_spins = input_spins[layout]
input_spins = input_spins.view(dim1, dim2)

print(input_spins)
