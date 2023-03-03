import numpy as np
import torch
from torch import nn

def create_model():
    NN = nn.Sequential(nn.Linear(784, 256, bias=True),
                   nn.ReLU(),
                   nn.Linear(256, 16, bias=True),
                   nn.ReLU(),
                   nn.Linear(16, 10, bias=True),
                   nn.ReLU())
    return NN

def count_parameters(model):
    sum = 0
    for param in model.parameters():
      sum += param.numel()
    return sum
