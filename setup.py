from setuptools import setup


install_requires=[
    'pandas',
    'matplotlib',
    'numpy>=1.13.3',
    'scikit-optimize',
    'scikit-learn',
    'tqdm',
    'tensorflow==1.13.1',
    'keras',
    'deap', # GA search
    # nas
    'gym',
    'networkx',
    'joblib',
    'mpi4py',
    'balsam',
]

setup(
  name = 'nas4candle',
  packages = ['nas4candle'],
  install_requires=install_requires,
  dependency_links=['https://github.com/balsam-alcf/balsam/tree/master#egg=balsam-0.2']
)
