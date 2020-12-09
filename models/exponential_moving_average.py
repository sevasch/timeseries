import torch
import torch.nn as nn

class ExponentialMovingAverage(nn.Module):
    def __init__(self, weight):
        super(ExponentialMovingAverage, self).__init__()
        self.weight = weight

    def forward(self, x):
        EMA = 0
        for i in range(x.shape[1]):
            EMA = self.weight * EMA + (1 - self.weight) * x[:, i]
        return EMA.unsqueeze(-1)

if __name__ == '__main__':
    x = torch.arange(10).float()
    m = ExponentialMovingAverage(0.1)
    print(x)
    print(m(x))