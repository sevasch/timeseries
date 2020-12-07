import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def split_sequence(x, n_past, n_future, step=1):
    sequences_past = []
    sequences_future = []
    for s in np.arange(0, len(x) - (n_past + n_future) + 1, step):
        sequences_past.append(x[s:s+n_past])
        sequences_future.append(x[s+n_past:s+n_past+n_future])
    return sequences_past, sequences_future

