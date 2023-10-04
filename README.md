# Multiprompt

## Package Dependencies

- cuda 11.3
- cu113
- pyG

## Running experiments

Due to the limited size, the default dataset is ENZYMES.  You need to change the corresponding parameters in *preprompt.py* and *execute.py* to train and evaluate on other datasets.

Pretrain:

- python pre_train.py

Prompt tune and test:

- python run.py
