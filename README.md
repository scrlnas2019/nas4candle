# nas4candle

For Theta at ALCF.

## Installation

```
$ module load cray-python/3.6.1.1
$ module load balsam/0.3
$ mkdir nas4candle-env
$ python -m venv --system-site-packages nas4candle-env
$ source nas4candle-env/bin/activate
$ git clone https://github.com/scrlnas2019/nas4candle.git
$ pip install --user -e nas4candle
```


Set your project name:
```
$ export PROJECT_NAME="project_name"
```

## Download data

For Combo:
```
$ python nas4candle/nas4candle/candle/Combo/combo_baseline_keras2.py
```

For NT3:
```
$ mkdir ~/data-tmp
$ python nas4candle/nas4candle/candle/NT3/nt3_baseline_keras2.py
```

For Uno:
```
$ mkdir ~/data-tmp
$ python nas4candle/nas4candle/candle/Uno/uno_baseline_keras2.py
```

## Balsam init

To create balsam db and start the db:
```
balsam init nas4candledb
source balsamactivate nas4candledb
```

To create balsam applications for A3C, A2C and RDM:
```
balsam app --name A3C --exe nas4candle/nasapi/search/nas/ppo_a3c_async.py
balsam app --name A2C --exe nas4candle/nasapi/search/nas/ppo_a3c_sync.py
balsam app --name RDM --exe nas4candle/nasapi/search/nas/nas_random.py
```

now if you run `balsam ls apps`, the expected output is:
```
pk | name |                 executable | description
----------------------------------------------------
1  | A3C  | python3.6 ppo_a3c_async.py |
2  | A2C  | python3.6 ppo_a3c_sync.py  |
3  | RDM  | python3.6 nas_random.py    |
```

## Combo

### Small search space

Async:
```
$ balsam job --name combo_async_small --workflow combo_async_small --app A3C --num-nodes 22 --args '--evaluator balsam --run nas4candle.candle.Combo.combo_baseline_keras2.run_model --problem nas4candle.candle.Combo.problems.problem_small.Problem'
$ balsam submit-launch --job-mode mpi -n 256 -t 360 -q default -A $PROJECT_NAME --wf-filter combo_async_small
```

Sync:
```
$ balsam job --name combo_sync_small --workflow combo_sync_small --app A2C --num-nodes 21 --args '--evaluator balsam --run nas4candle.candle.Combo.combo_baseline_keras2.run_model --problem nas4candle.candle.Combo.problems.problem_small.Problem'
$ balsam submit-launch --job-mode mpi -n 256 -t 360 -q default -A $PROJECT_NAME --wf-filter combo_sync_small
```

Random:
```
$ balsam job --name combo_rdm_small --workflow combo_rdm_small --app RDM --num-nodes 22 --args '--evaluator balsam --run nas4candle.candle.Combo.combo_baseline_keras2.run_model --problem nas4candle.candle.Combo.problems.problem_small.Problem'
$ balsam submit-launch --job-mode mpi -n 256 -t 360 -q default -A $PROJECT_NAME --wf-filter combo_rdm_small
```

### Large search space

Async:
```
$ balsam job --name combo_async_large --workflow combo_async_large --app A3C --num-nodes 22 --args '--evaluator balsam --run nas4candle.candle.Combo.combo_baseline_keras2.run_model --problem nas4candle.candle.Combo.problems.problem_large.Problem'
$ balsam submit-launch --job-mode mpi -n 256 -t 360 -q default -A $PROJECT_NAME --wf-filter combo_async_large
```

Sync:
```
$ balsam job --name combo_sync_large --workflow combo_sync_large --app A2C --num-nodes 21 --args '--evaluator balsam --run nas4candle.candle.Combo.combo_baseline_keras2.run_model --problem nas4candle.candle.Combo.problems.problem_large.Problem'
$ balsam submit-launch --job-mode mpi -n 256 -t 360 -q default -A $PROJECT_NAME --wf-filter combo_sync_large
```

Random:
```
$ balsam job --name combo_rdm_large --workflow combo_rdm_large --app RDM --num-nodes 22 --args '--evaluator balsam --run nas4candle.candle.Combo.combo_baseline_keras2.run_model --problem nas4candle.candle.Combo.problems.problem_large.Problem'
$ balsam submit-launch --job-mode mpi -n 256 -t 360 -q default -A $PROJECT_NAME --wf-filter combo_rdm_large
```

## Uno

### Small search space

Async:
```
$ balsam job --name uno_async_large --workflow uno_async_large --app A3C --num-nodes 22 --args '--evaluator balsam --run nas4candle.candle.Uno.uno_baseline_keras2.run_model --problem nas4candle.candle.Uno.problems.problem_small.Problem'
$ balsam submit-launch --job-mode mpi -n 256 -t 360 -q default -A $PROJECT_NAME --wf-filter uno_async_large
```

### Large search space

Async:
```
$ balsam job --name uno_async_large --workflow uno_async_large --app A3C --num-nodes 22 --args '--evaluator balsam --run nas4candle.candle.Uno.uno_baseline_keras2.run_model --problem nas4candle.candle.Uno.problems.problem_large.Problem'
$ balsam submit-launch --job-mode mpi -n 256 -t 360 -q default -A $PROJECT_NAME --wf-filter uno_async_large
```

## NT3

### Small search space

Async:
```
$ balsam job --name nt3_async_small --workflow nt3_async_small --app A3C --num-nodes 22 --args '--evaluator balsam --run nas4candle.nasapi.search.nas.model.run.alpha.run --problem nas4candle.candle.NT3.problems.problem_small.Problem'
$ balsam submit-launch --job-mode mpi -n 256 -t 360 -q default -A $PROJECT_NAME --wf-filter nt3_async_small
```