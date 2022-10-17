import numpy as np
import torch
from matplotlib import pyplot as plt

def fanin_init(size, fanin=None):
    # 一种比较合理的初始化网络参数https://arxiv.org/abs/1502.01852
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    x = torch.Tensor(size).uniform_(-v, v)
    return x.type(torch.FloatTensor)

