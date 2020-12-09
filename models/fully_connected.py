import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, n_past):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(features_in := n_past, features_in//2), nn.ReLU())
        self.fc2 = nn.Linear(features_in//2, 1)

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(fc1_out)
        return fc2_out


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_past = 730
    n_future = 1
    step = 1

    torch.manual_seed(0)

    # create model, loss function and optimizer
    model = FullyConnected(n_past).cuda()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # load data
    import pandas
    import numpy as np
    df = pandas.read_csv('../data/gold.csv')
    x = np.array(df['Price'])[:8000]

    from data.single_series_dataset import SingleSeriesDataset
    from torch.utils.data import DataLoader
    subset_train = SingleSeriesDataset(x[:5000], n_past=n_past, n_future=n_future, step=step)
    subset_val = SingleSeriesDataset(x[5000:], n_past=n_past, n_future=n_future, step=step)

    # create dataloaders
    loader_train = torch.utils.data.DataLoader(subset_train, batch_size=2, shuffle=True)
    loader_val = torch.utils.data.DataLoader(subset_val, batch_size=2, shuffle=True)
    print('training samples: {}, validation samples: {}'.format(len(loader_train), len(loader_val)))

    # train
    from training import train_epochs
    ep_loss_train, ep_loss_val = train_epochs(model, optimizer, loss_fn,
                                             loader_train, loader_val,
                                             n_epochs=80,
                                             log_project='timeseries', log_description='fc_deleteme', log_wandb=False)

    # plot losses
    plt.plot(ep_loss_train, label='train')
    plt.plot(ep_loss_val, label='val')
    plt.grid(True)
    plt.legend()
    plt.show()

