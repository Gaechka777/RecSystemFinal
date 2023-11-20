import torch
import torch.nn.functional as F

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}