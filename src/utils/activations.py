import torch
import torch.nn as nn

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def tanh(x):
    return ( torch.exp(x) - torch.exp(-x) ) / ( torch.exp(x) + torch.exp(-x) )

def relu(x):
    return torch.max(torch.tensor(0.0), x)

def leaky_relu(x, alpha=0.01):
    return torch.where(x>0,x,x*alpha)

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def gelu(x):
    # 将输入沿指定维度分成两部分
    a,b = x.chunk(2,-1)
    return a * sigmoid(b)

def swiglu(x):
    # 将输入沿指定维度分成两部分
    a,b = x.chunk(2,-1)
    return a * swish(b)

