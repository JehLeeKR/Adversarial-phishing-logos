import torch
from torch import nn

class CrossEntropyLossWithThreshold(nn.Module):
    def __init__(self, threshold=0.5):
        super(CrossEntropyLossWithThreshold, self).__init__()
        self.threshold = threshold

    def forward(self, output, target):
        output = torch.softmax(output, dim=-1)
        target = torch.clamp(target, min=None, max=self.threshold)
        target = torch.softmax(target, dim=-1)
        return self.cross_entropy(output, target)

    def cross_entropy(self, input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))