import torch
from torch import nn
import torch.nn.functional as F

class SiameseLoss(nn.Module):
    def __init__(self):
        super(SiameseLoss, self).__init__()

    def forward(self, sim_list):
        return torch.mean(torch.tensor(sim_list))