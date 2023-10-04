import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import process
import os

dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True
nonlinearity = 'prelu'  # special name to separate parameters

adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
features, _ = process.preprocess_features(features)

# lals=torch.argmax(labels)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

lals = torch.argmax(labels[0:labels.shape[0]], dim=2)
list = [0] * 1500
os.makedirs("fewshot")
trainlal = torch.FloatTensor(1, 7)
trainfeature = torch.FloatTensor(7,)
start = 0
for i in range(0, 99):
    cnt = 0
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    for j in range(0, 1500):
        if cnt == 7:
            # os.removedir("fewshot/{}".format(i))
            os.makedirs("fewshot/{}".format(i))
            torch.save(trainlal, "fewshot/{}/labels.pt".format(i))
            torch.save(trainfeature, "fewshot/{}/idx.pt".format(i))
            print('number', i, 'trainlal', trainlal, '\n')
            print('trainfeature', trainfeature, '\n')
            break


        if lals[0][j].item() == 0 and cnt0 == 0 and list[j] == 0:
            trainlal[0][cnt] = 0
            trainfeature[cnt] = j
            list[j] = 1
            print(j,' ,')
            cnt = cnt + 1
            cnt0 = 1
        if lals[0][j].item() == 1 and cnt1 == 0 and list[j] == 0:
            trainlal[0][cnt] = 1
            trainfeature[cnt] = j
            list[j] = 1
            print(j,' ,')
            cnt = cnt + 1
            cnt1 = 1
        if lals[0][j].item() == 2 and cnt2 == 0 and list[j] == 0:
            trainlal[0][cnt] = 2
            trainfeature[cnt] = j
            list[j] = 1
            print(j,' ,')
            cnt = cnt + 1
            cnt2 = 1
        if lals[0][j].item() == 3 and cnt3 == 0 and list[j] == 0:
            trainlal[0][cnt] = 3
            trainfeature[cnt] = j
            list[j] = 1
            print(j,' ,')
            cnt = cnt + 1
            cnt3 = 1
        if lals[0][j].item() == 4 and cnt4 == 0 and list[j] == 0:
            trainlal[0][cnt] = 4
            trainfeature[cnt] = j
            list[j] = 1
            print(j, ' ,')
            cnt = cnt + 1
            cnt4 = 1
        if lals[0][j].item() == 5 and cnt5 == 0 and list[j] == 0:
            trainlal[0][cnt] = 5
            trainfeature[cnt] = j
            list[j] = 1
            print(j, ' ,')
            cnt = cnt + 1
            cnt5 = 1
        if lals[0][j].item() == 6 and cnt6 == 0 and list[j] == 0:
            trainlal[0][cnt] = 6
            trainfeature[cnt] = j
            list[j] = 1
            print(j, ' ,')
            cnt = cnt + 1
            cnt6 = 1
print("end")
