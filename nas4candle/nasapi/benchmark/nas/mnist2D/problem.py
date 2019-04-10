# TODO : not ready

from nas4candle.nasapi.search.nas.contrib.google_nas_net import create_structure
from nas4candle.nasapi.benchmark.nas.mnist2D.load_data import load_data
from nas4candle.nasapi.benchmark import Problem

Problem = Problem()
Problem.add_dim('regression', False)
Problem.add_dim('load_data', {
    'func': load_data
})
Problem.add_dim('create_structure', {
    'func': create_structure,
    'kwargs': {}
})
Problem.add_dim('hyperparameters', {
    'batch_size': 100,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'num_epochs': 50,
    'loss_metric': 'mean_softmax_cross_entropy',
    'test_metric': 'accuracy'
})


if __name__ == '__main__':
    print(Problem)
