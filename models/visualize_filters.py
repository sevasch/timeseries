import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_filters(state_dict: dict):
    layer_names = []
    layer_weights = []
    n_filters_max = 0
    for key, value in state_dict.items():
        if key.split('.')[-1] == 'weight':
            layer_names.append(key)
            layer_weights.append(value)
            n_filters_max = max(n_filters_max, value.shape[0])

    fig, axes = plt.subplots(nrows=len(layer_names), ncols=n_filters_max)

    # show weights
    for layer_no, weight in enumerate(layer_weights):
        for filter_no, filter in enumerate(weight):
            filter = filter.unsqueeze(-1) if filter.ndim == 1 else filter
            filter = filter.cpu().numpy()
            axes[layer_no, filter_no].imshow(filter, cmap='gray')

    # disable all axis and
    for col_no in range(n_filters_max):
        for ax, name in zip(axes[:, col_no], layer_names):
            if col_no == 0:
                ax.set_ylabel(name, rotation=90, size='small')
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    plt.show()




if __name__ == '__main__':
    from models.lstm_seq2seq import LSTMSeq2Seq
    ckpt = torch.load('/home/sebastian/PycharmProjects/stockfun/models/saved_models/20201113_2141_5-1_features_res/iter192_best.pt')
    model = ckpt['model']
    model.load_state_dict(ckpt['model_state_dict'])

    visualize_filters(ckpt['model_state_dict'])
