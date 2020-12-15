import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.single_series_dataset import SingleSeriesDataset
import matplotlib.pyplot as plt

def relative_to_abs(past_abs, future_change_rel):
    future_abs = torch.zeros_like(future_change_rel)
    last_abs = past_abs[:, -1].clone()
    for i in range(future_change_rel.shape[1]):
        future_abs[:, i] = last_abs + future_change_rel[:, i] * last_abs
        last_abs = future_abs[:, i]
    return future_abs

def change_to_abs(past_abs, future_change_abs):
    future_abs = torch.zeros_like(future_change_abs)
    last_abs = past_abs[:, -1].clone()
    for i in range(future_change_abs.shape[1]):
        future_abs[:, i] = last_abs + future_change_abs[:, i]
        last_abs = future_abs[:, i]
    return future_abs

def evaluate_series(model, dataloader):
    predictions = []
    predictions_abs = []
    L2_errors = []
    for sample in tqdm(dataloader, total=len(dataloader)):
        predictions.append(prediction := model(sample['input'].cuda(), sample['target'].shape[1]))
        predictions_abs.append(prediction_abs := change_to_abs(sample['past'], prediction.cpu()))
        L2_errors.append(torch.sqrt(torch.abs(prediction_abs - sample['future'])))
    return predictions_abs, predictions, L2_errors

if __name__ == '__main__':
    # define model
    from models.exponential_moving_average import ExponentialMovingAverage
    from models.multi_predict_wrapper import MultiPredictWrapper
    # m = ExponentialMovingAverage(0.1)
    from models.fully_connected import FullyConnected
    from models.filters_fully_connected import FiltersFullyConnected
    ckpt = torch.load('/home/sebastian/PycharmProjects/timeseries/models/saved_models/20201211_1908_ffc_deleteme16/best.pt')
    m = ckpt['model']
    m.load_state_dict(ckpt['model_state_dict'])
    m.cuda()
    m = MultiPredictWrapper(m)

    # create data
    n = np.linspace(0, 10, 300)
    sequence = np.concatenate([np.linspace(0, 10, 100), 2*np.linspace(0, 10, 100), 0.5*np.linspace(0, 10, 100)])

    # evaluation
    n_past = 16
    n_future = 20
    step = 30

    # create datasets
    dataset = SingleSeriesDataset(sequence, n_past, n_future, step)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # evaluate
    predictions_abs, predictions, errors = evaluate_series(m, loader)

    # plot
    fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex='col', sharey='row')
    ax[0].set_title('absolute values'), ax[1].set_title('change'), ax[2].set_title('L2 errors')
    ax[0].grid(True), ax[1].grid(True), ax[2].grid(True)

    p00 = ax[0].plot(sequence, '-', c='b', label='series')
    p10 = ax[1].plot(torch.cat((torch.zeros(1), torch.tensor((sequence[1:] - sequence[:-1])))), '-', c='b', label='series')
    for i, (sample, prediction_abs, prediction, error) in enumerate(zip(loader, predictions_abs, predictions, errors)):
        prediction_abs_padded = len(sequence) * [torch.tensor(float('nan'))]
        prediction_padded = len(sequence) * [torch.tensor(float('nan'))]
        error_padded = len(sequence) * [torch.tensor(float('nan'))]
        prediction_abs_padded[step*i+n_past:step*i+n_past+n_future] = prediction_abs[0]
        prediction_padded[step*i+n_past:step*i+n_past+n_future] = prediction[0]
        error_padded[step*i+n_past:step*i+n_past+n_future] = error[0]
        p01 = ax[0].plot(prediction_abs_padded, '-', c='r', linewidth=1, label='prediction')
        p11 = ax[1].plot(prediction_padded, '-', c='r', linewidth=1, label='prediction')
        p2 = ax[2].plot(error_padded, '-', c='b', linewidth=1, label='errors')
    ax[0].legend((p00[0], p01[0]), ('series', 'predictions'))
    ax[1].legend((p10[0], p11[0]), ('series', 'predictions'))
    plt.suptitle(m._get_name())
    print('Mean L2 error: {}'.format(torch.mean(torch.tensor([torch.mean(e) for e in errors]))))
    plt.show()

