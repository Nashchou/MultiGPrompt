# MultiGPrompt

## Description

The repository is organised as follows:

- **data/**: contains data we use.
- **modelset/**: contains pre-trained model we use.
- **MutilGPrompt_CoraCiteseer_node/**: implements pre-training and downstream tasks for Cora and Citeseer.
- **MutilGPrompt_TU_node/**: implements pre-training and downstream task for ENZYMES and PROTEINS. 
- **MutilGPrompt_TU_graph/**: implements pre-training for BZR and COX2, downstream task for BZR,COX2,ENZYMES,PROTEINS.

## Package Dependencies

- python 3.8.16
- pytorch 1.10.1
- cuda 11.3
- pyG 2.0.0

## Running experiments

Due to the limitation of file size, we upload all datasets except for PROTEINS. But the pre-trained model for each datasets is uploaded. 

### Node Classification for Cora and Citeseer 
Default dataset is Cora. You need to change the corresponding parameters in *preprompt.py*, *downprompt.py.py* and *execute.py* to train and evaluate on other datasets.

Pretrain Prompt tune and test:
`python execute.py`

### Node Classification for ENZYMES and PROTEINS 
Default dataset is ENZYMES. You need to change the corresponding parameters in *preprompt.py*, *downprompt.py.py* and *execute.py* to train and evaluate on other datasets.

Pretrain Prompt tune and test:
`python execute.py`


### Graph Classification
Default dataset is ENZYMES. You need to change the corresponding parameters in *preprompt.py*, *downprompt.py.py* and *execute.py* to train and evaluate on other datasets.

Pretrain Prompt tune and test:
`python execute.py`


