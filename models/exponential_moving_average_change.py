import torch
import torch.nn as nn

class ExponentialMovingAverageChange(nn.Module):
    def __init__(self, weight):
        super(ExponentialMovingAverageChange, self).__init__()
        self.weight = weight

    def forward(self, x):
        change = x[1:] - x[:-1]
        EMA = 0
        for value in change:
            EMA = self.weight * EMA + (1 - self.weight) * value
        return x[-1] + EMA

if __name__ == '__main__':
    x = torch.arange(10).float()
    m = ExponentialMovingAverageChange(0.1)
    print(x)
    print(m(x))