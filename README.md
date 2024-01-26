# MultiGPrompt
We provide the code (in pytorch) and datasets for our paper [**"MultiGPrompt for Multi-Task Pre-Training and Prompting on Graphs"**](https://arxiv.org/pdf/2312.03731.pdf), 
which is accepted by WWW2024.
## Description

The repository is organised as follows:

- **data/**: contains data we use.
- **modelset/**: contains pre-trained model we use.
- **MutilGPrompt_CoraCiteseer_node/**: implements pre-training and node level downstream tasks for Cora and Citeseer.
- **MutilGPrompt_TU_node/**: implements pre-training and node level downstream task for ENZYMES and PROTEINS. 
- **MutilGPrompt_TU_graph/**: implements pre-training for BZR and COX2, graph level downstream task for BZR,COX2,ENZYMES,PROTEINS.

## Package Dependencies

- python 3.8.16
- pytorch 1.10.1
- cuda 11.3
- pyG 2.0.0

## Running experiments

Due to the limitation of file size, we upload all datasets except for PROTEINS. But the pre-trained model for each datasets is uploaded. 

### Node Classification for Cora and Citeseer 
Default dataset is Cora. You need to change the corresponding parameters in *preprompt.py*, *downprompt.py.py* and *execute.py* to train and evaluate on other datasets.

Pretrain and Prompt tune:
`python execute.py`

### Node Classification for ENZYMES and PROTEINS 
Default dataset is ENZYMES. You need to change the corresponding parameters in *preprompt.py*, *downprompt.py.py* and *execute.py* to train and evaluate on other datasets.

Pretrain and Prompt tune:
`python execute.py`


### Graph Classification for BZR,COX2,ENZYMES,PROTEINS.
Default dataset is ENZYMES. You need to change the corresponding parameters in *preprompt.py*, *downprompt.py.py* and *execute.py* to train and evaluate on other datasets.

Pretrain and Prompt tune:
`python execute.py`


