# nas4candle

## Installation

```
cd nas4candle
pip install -e .
```

## Balsam init

Create balsam db and start the db.
```
balsam init nasdb
source balsamactivate nasdb
```

Create balsam applications for A3C, A2C and RDM.
```
balsam app --name A3C --exe nas4candle/nasapi/search/nas/ppo_a3c_async.py
balsam app --name A2C --exe nas4candle/nasapi/search/nas/ppo_a3c_sync.py
balsam app --name RDM --exe nas4candle/nasapi/search/nas/nas_random.py
```

## Combo

## Uno

## NT3