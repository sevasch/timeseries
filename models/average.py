import torch
import torch.nn as nn

class Average(nn.Module):
    def __init__(self):
        super(Average, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)

if __name__ == '__main__':
    x = torch.arange(10).float()
    m = Average()
    print(x)
    print(m(x))