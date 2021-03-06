"""
Benchmarks are here for you to test the performance of different search algorithm and reproduce our results. They can also help you to test your installation of nas4candle.nasapi or
discover the many parameters of a search. In nas4candle.nasapi we have two different kind of benchmark. The first type is `hyper parameters search` benchmark and the second type is  `neural architecture search` benchmark. To see a full explanation about the different kind of search please refer to the following pages: `Hyperparameter Searchs <search/hps/index.html>`_  & :ref:`available-nas-benchmarks`.
"""

from nas4candle.nasapi.benchmark.problem import Problem, HpProblem

__all__ = ['Problem', 'HpProblem']
