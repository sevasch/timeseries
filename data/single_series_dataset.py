import torch
from torch.utils.data import Dataset

class SingleSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, sequence, n_past, n_future, step=1):
        super(SingleSeriesDataset, self).__init__()
        self.n_past = n_past
        self.n_future = n_future

        self.sequences = []
        for s in torch.arange(0, len(sequence) - (n_past + n_future) + 1, step):
            self.sequences.append(torch.tensor(sequence[s:s + n_past + n_future]).float())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # store original data
        sample = {}
        sample['past'] = self.sequences[idx][:self.n_past]
        sample['future'] = self.sequences[idx][-self.n_future:]

        # transform data for training
        sequence = self.sequences[idx].clone()

        # calculate relative change
        sequence = torch.cat((torch.zeros(1), (sequence[1:] - sequence[:-1]) / sequence[:-1]))

        # split into past and future

        sample['input'] = sequence[:self.n_past]
        sample['target'] = sequence[-self.n_future:]
        return sample

if __name__ == '__main__':
    torch.manual_seed(0)
    # load data
    import pandas
    import matplotlib.pyplot as plt
    import numpy as np
    df = pandas.read_csv('gold.csv')
    x = np.array(df['Price'])

    # create dataset and loader
    dataset = SingleSeriesDataset(x, n_past=5, n_future=5, step=1)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    # visualize
    for i, sample in enumerate(loader):
        p1 = plt.plot(sample['past'][0])
        p2 = plt.plot(torch.cat((torch.tensor(len(sample['input'][0]) * [torch.tensor(float('nan'))]), sample['target'][0])), c=p1[0].get_color())
        if i > 1:
            break
    plt.grid(True)
    plt.show()