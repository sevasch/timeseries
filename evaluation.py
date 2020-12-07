import torch
from tqdm import tqdm
from data.tools import split_sequence
import matplotlib.pyplot as plt

def evaluate_series(model, sequences_past, sequences_future):
    predictions = []
    L2_errors = []
    for sequence_past, sequence_future in tqdm(zip(sequences_past, sequences_future), total=len(sequences_past)):
        predictions.append(prediction := model(sequence_past, len(sequence_future)))
        L2_errors.append(torch.sqrt(torch.abs(prediction - sequence_future)))
    return predictions, L2_errors

if __name__ == '__main__':
    # define model
    from models.exponential_moving_average_change import ExponentialMovingAverageChange
    from models.multi_predict_wrapper import MultiPredictWrapper
    m = ExponentialMovingAverageChange(0.5)
    # m = SARIMAXLib()
    m = MultiPredictWrapper(m)

    # make data
    import pandas
    df = pandas.read_csv('data/gold.csv')
    x = torch.tensor(df['Price'])[-500:]
    # x = torch.sin(torch.linspace(0, 10, 100).float())

    # evaluation
    sequences_past, sequences_future = split_sequence(x, 100, 10, step:=10)
    predictions, errors = evaluate_series(m, sequences_past, sequences_future)

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex='col', sharey='row')
    ax[0].set_title('series and predictions'), ax[1].set_title('L2 errors')
    ax[0].grid(True), ax[1].grid(True)

    p0 = ax[0].plot(x, '-', c='b', label='series')
    for i, (sequence_past, sequence_future, prediction, error) in enumerate(zip(sequences_past, sequences_future, predictions, errors)):
        prediction_padded = len(x) * [torch.tensor(float('nan'))]
        error_padded = len(x) * [torch.tensor(float('nan'))]
        prediction_padded[step*i+len(sequence_past):step*i+len(sequence_past)+len(prediction)] = prediction
        error_padded[step*i+len(sequence_past):step*i+len(sequence_past)+len(prediction)] = error
        p1 = ax[0].plot(prediction_padded, '-', c='r', linewidth=1, label='prediction')
        p2 = ax[1].plot(error_padded, '-', c='b', linewidth=1, label='errors')
    ax[0].legend((p0[0], p1[0]), ('series', 'predictions'))
    plt.suptitle(m._get_name())
    print('Mean L2 error: {}'.format(torch.mean(torch.tensor([torch.mean(e) for e in errors]))))
    plt.show()

