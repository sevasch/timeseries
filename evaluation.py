import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.single_series_dataset import SingleSeriesDataset
import matplotlib.pyplot as plt

def relative_to_abs(past_abs, future_rel):
    future_abs = torch.zeros_like(future_rel)
    last_abs = past_abs[:, -1].clone()
    for i in range(future_rel.shape[1]):
        future_abs[:, i] = last_abs + future_rel[:, i] * last_abs
        last_abs = future_abs[:, i]
    return future_abs

def evaluate_series(model, dataloader):
    predictions_rel = []
    predictions_abs = []
    L2_errors = []
    for sample in tqdm(dataloader, total=len(dataloader)):
        predictions_rel.append(prediction_rel := model(sample['input'].cuda(), sample['target'].shape[1]))
        predictions_abs.append(prediction_abs := relative_to_abs(sample['past'], prediction_rel.cpu()))
        L2_errors.append(torch.sqrt(torch.abs(prediction_abs - sample['future'])))
    return predictions_abs, predictions_rel, L2_errors

if __name__ == '__main__':
    # define model
    from models.exponential_moving_average import ExponentialMovingAverage
    from models.multi_predict_wrapper import MultiPredictWrapper
    # m = ExponentialMovingAverage(0.5)
    # m = SARIMAXLib()
    from models.fully_connected import FullyConnected
    ckpt = torch.load('/home/sebastian/PycharmProjects/timeseries/models/saved_models/20201209_1947_fc_deleteme/best.pt')
    ckpt = torch.load('/home/sebastian/PycharmProjects/timeseries/models/saved_models/20201209_2005_fc_deleteme/best.pt')
    m = ckpt['model']
    m.load_state_dict(ckpt['model_state_dict'])
    m.cuda()
    m = MultiPredictWrapper(m)

    # load data
    import pandas
    df = pandas.read_csv('data/gold.csv')
    x = np.array(df['Price'])[-3000:]
    # x = torch.sin(torch.linspace(0, 10, 100).float())

    # evaluation
    n_past = 730
    n_future = 5
    step = 50
    dataset = SingleSeriesDataset(x, n_past=n_past, n_future=n_future, step=step)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions_abs, predictions_rel, errors = evaluate_series(m, loader)

    # plot
    fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex='col', sharey='row')
    ax[0].set_title('absolute values'), ax[1].set_title('relative change'), ax[2].set_title('L2 errors')
    ax[0].grid(True), ax[1].grid(True), ax[2].grid(True)

    p00 = ax[0].plot(x, '-', c='b', label='series')
    p10 = ax[1].plot(torch.cat((torch.zeros(1), torch.tensor((x[1:] - x[:-1]) / x[:-1]))), '-', c='b', label='series')
    for i, (sample, prediction_abs, prediction_rel, error) in enumerate(zip(loader, predictions_abs, predictions_rel, errors)):
        prediction_abs_padded = len(x) * [torch.tensor(float('nan'))]
        prediction_rel_padded = len(x) * [torch.tensor(float('nan'))]
        error_padded = len(x) * [torch.tensor(float('nan'))]
        prediction_abs_padded[step*i+n_past:step*i+n_past+n_future] = prediction_abs[0]
        prediction_rel_padded[step*i+n_past:step*i+n_past+n_future] = prediction_rel[0]
        error_padded[step*i+n_past:step*i+n_past+n_future] = error[0]
        p01 = ax[0].plot(prediction_abs_padded, '-', c='r', linewidth=1, label='prediction')
        p11 = ax[1].plot(prediction_rel_padded, '-', c='r', linewidth=1, label='prediction')
        p2 = ax[2].plot(error_padded, '-', c='b', linewidth=1, label='errors')
    ax[0].legend((p00[0], p01[0]), ('series', 'predictions'))
    ax[1].legend((p10[0], p11[0]), ('series', 'predictions'))
    plt.suptitle(m._get_name())
    print('Mean L2 error: {}'.format(torch.mean(torch.tensor([torch.mean(e) for e in errors]))))
    plt.show()

