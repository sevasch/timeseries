import torch

def split_sequence(sequence, n_timesteps, step):
    sequences = []
    for s in torch.arange(0, len(sequence) - n_timesteps + 1, step):
        sequences.append(torch.tensor(sequence[s:s + n_timesteps]).float())