# Multiprompt

## Description

The repository is organised as follows:

- **data/**: contains data we use.
- **modelset/**: contains pre-train model we use
- **Mutilprompt_CoraCiteseer_node/**: implements pre-training and downstream tasks at the node level for Cora and Citeseer.
- **Mutilprompt_TU_node/**: implements pre-training and downstream tasks at the node level for TUdatasets. 
- **Mutilprompt_TU_graph/**: implements pre-training and downstream tasks at the graph level for TUdatasets.

## Package Dependencies

- cuda 11.3
- cu113
- pyG

## Running experiments

Due to the limitation of file size, we upload all datasets except for PROTEINS.But the pre-trained model for each datasets is uploaded. You need to change the corresponding parameters in *preprompt.py*,*downprompt.py.py*and *execute.py* to train and evaluate on other datasets.

