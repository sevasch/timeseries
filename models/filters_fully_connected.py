import torch
import torch.nn as nn

class FiltersFullyConnected(nn.Module):
    def __init__(self, n_past):
        super(FiltersFullyConnected, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(4, 8, kernel_size=7, stride=2, padding=3),
                                   nn.MaxPool1d(kernel_size=5, stride=1, padding=2),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(8, 16, kernel_size=9, stride=4, padding=4),
                                   nn.ReLU())

        features_in = 16 * n_past//2//4
        self.fc1 = nn.Sequential(nn.Linear(features_in, features_in//2), nn.ReLU())
        self.fc2 = nn.Linear(features_in//2, 1)

    def forward(self, x):
        conv1_out = self.conv1(x.unsqueeze(-2))
        # print(conv1_out.shape)
        conv2_out = self.conv2(conv1_out)
        # print(conv2_out.shape)
        conv3_out = self.conv3(conv2_out)
        # print(conv3_out.shape)
        fc1_out = self.fc1(conv3_out.flatten(1, 2))
        fc2_out = self.fc2(fc1_out)
        return fc2_out



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_past = 16
    n_future = 1
    step = 1

    torch.manual_seed(1)

    # create model, loss function and optimizer
    model = FiltersFullyConnected(n_past).cuda()
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # create dataset
    import numpy as np
    n = np.linspace(0, 10, 100)
    sequences = []
    for _ in range(100):
        sequences.append(np.random.randn() * n + np.random.randn())
    # for _ in range(100):
    #     sequences.append(np.random.randn() * np.sin(np.random.randn() * n + np.random.randn()) + np.random.randn())

    # shuffle the datasets
    import random
    random.shuffle(sequences)

    from data.single_series_dataset import SingleSeriesDataset
    from torch.utils.data import DataLoader
    datasets_train = []
    for sequence in sequences[:len(sequences)//3*2]:
        datasets_train.append(SingleSeriesDataset(sequence, n_past, n_future, noise=0.))
    datasets_val = []
    for sequence in sequences[len(sequences)//3*2:]:
        datasets_val.append(SingleSeriesDataset(sequence, n_past, n_future))
    subset_train = torch.utils.data.ConcatDataset(datasets_train)
    subset_val = torch.utils.data.ConcatDataset(datasets_val)

    print('training samples: {}, validation samples: {}'.format(len(subset_train), len(subset_val)))

    # create dataloaders
    loader_train = torch.utils.data.DataLoader(subset_train, batch_size=1, shuffle=True)
    loader_val = torch.utils.data.DataLoader(subset_val, batch_size=20, shuffle=False)

    # for sample in loader_train:
    #     plt.subplot(2, 1, 1), plt.plot(sample['past_augmented'][0])
    #     plt.subplot(2, 1, 1), plt.plot(sample['past'][0])
    #     plt.subplot(2, 1, 2), plt.plot(sample['input'][0]),plt.show()

    # train
    from training import train_iterative
    loss_train, loss_val = train_iterative(model, optimizer, loss_fn,
                                           loader_train, loader_val,
                                           n_iterations=100, n_steps_per_iteration=1000,
                                           log_project='timeseries', log_description='ffc_deleteme' + str(n_past), log_wandb=False)

    # plot losses
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.show()

