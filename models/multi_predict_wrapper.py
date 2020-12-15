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
        predictions = torch.zeros(x.shape[0], n_future)
        for i in range(n_future):
            prediction = self.model(x_new[:, -x.shape[1]:])
            x_new = torch.cat((x_new[:, 1:], prediction), dim=1)
            predictions[:, i] = prediction

        return predictions