import torch
import torch.nn as nn

class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target, gamma=2.0):
        mask = (gamma * target) / (target + 10e-8)
        self.loss = self.criterion(input * mask, target * mask)
        return self.loss
