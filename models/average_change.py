import torch
import torch.nn as nn

class AverageChange(nn.Module):
    def __init__(self):
        super(AverageChange, self).__init__()

    def forward(self, x):
        change = x[1:] - x[:-1]
        return (x[-1] + torch.mean(change)).float()

if __name__ == '__main__':
    x = torch.arange(10)
    m = AverageChange()
    print(x)
    print(m(x))