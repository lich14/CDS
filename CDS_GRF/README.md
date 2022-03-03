
## Run an experiment 

```shell
python3 main.py --config=CDS_QMIX --env-config=academy_3_vs_1_with_keeper 
```

The config files act as defaults for an algorithm or environment. 

They are all located in `config`.

`--config` refers to the config files in `config/algs`, one can choose CDS_QMIX or CDS_QPLEX (CDS + QMIX is recommended for GRF)

`--env-config` refers to the config files in `config/envs`, one can choose academy_3_vs_1_with_keeper or academy_counterattack_hard

## Data

GRF data is located in 'data', where we show the performance of CDS with QPLEX or QMIX.

## Scenario
We delete agents initialized in our half in the 'academy counterattack hard' scenario, which is located in 'data/academy_counterattack_hard.py'.
