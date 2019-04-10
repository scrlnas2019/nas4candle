"""
The ``search`` module brings a modular way to implement new search algorithms and two sub modules. One is for hyperparameter search ``nas4candle.nasapi.search.hps`` and one is for neural architecture search ``nas4candle.nasapi.search.nas``.
The ``Search`` class is abstract and has different subclasses such as: ``nas4candle.nasapi.search.ambs`` and ``nas4candle.nasapi.search.ga``.
"""

from nas4candle.nasapi.search.search import Search

__all__ = ['Search']
