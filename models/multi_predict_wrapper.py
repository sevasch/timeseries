'''
Wrapper class to predict n_future values into the future for models capable of
predicting only one value into the future.
'''

import torch
import torch.nn as nn

class MultiPredictWrapper(nn.Module):
    def __init__(self, model):
        super(MultiPredictWrapper, self).__init__()
        self.model = model

    def forward(self, x, n_future):
        x_new = x.clone()
        for _ in range(n_future):
            prediction = self.model(x_new[-len(x):])
            x_new = torch.cat((x_new, prediction.clone().unsqueeze(0)), dim=0)
        return x_new[-n_future:]