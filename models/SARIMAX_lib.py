import torch
import torch.nn as nn
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXLib(nn.Module):
    def __init__(self):
        super(SARIMAXLib, self).__init__()

    def forward(self, x):
        model = SARIMAX(x.numpy(), order=(2, 1, 2), seasonal_order=(2, 1, 2, 40), trend='c',
                        enforce_stationarity=False, enforce_invertibility=False)
        # fit model
        model_fit = model.fit(disp=False)

        # make one step forecast
        yhat = model_fit.predict(len(x), len(x))

        return torch.from_numpy(yhat).squeeze()


if __name__ == '__main__':
    x = torch.arange(10).float()
    m = SARIMAXLib()
    print(x)
    print(m(x))