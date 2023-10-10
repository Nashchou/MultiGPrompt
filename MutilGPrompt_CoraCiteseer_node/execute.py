import numpy as np
import scipy.sparse as sp

import random
from models import LogReg
from preprompt import PrePrompt
import preprompt
from utils import process
import pdb
import aug
import os
import argparse
from downprompt import downprompt,featureprompt
import csv
parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset', type=str, default="cora", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--save_name', type=str, default='modelset/cora.pkl', help='save ckpt name')

args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = args.seed
random.seed(seed)
np.random.seed(seed)
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

batch_size = 1
nb_epochs = 1000
patience = 20
lr = 0.0001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
sparse = True
useMLP = False

nonlinearity = 'prelu'  # special name to separate parameters
adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)  

features, _ = process.preprocess_features(features)
negetive_sample = preprompt.prompt_pretrain_sample(adj,200)

nb_nodes = features.shape[0]  # node number
ft_size = features.shape[1]  # node features dim
nb_classes = labels.shape[1]  # classes = 6

features = torch.FloatTensor(features[np.newaxis])

'''
# ------------------------------------------------------------
# edge node mask subgraph
# ------------------------------------------------------------
'''
# print("Begin Aug:[{}]".format(args.aug_type))
# if args.aug_type == 'edge':

aug_features1edge = features
aug_features2edge = features

aug_adj1edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
aug_adj2edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges


aug_features1mask = aug.aug_random_mask(features, drop_percent=drop_percent)
aug_features2mask = aug.aug_random_mask(features, drop_percent=drop_percent)

aug_adj1mask = adj
aug_adj2mask = adj

'''
# ------------------------------------------------------------
'''

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))

aug_adj1mask = process.normalize_adj(aug_adj1mask + sp.eye(aug_adj1mask.shape[0]))
aug_adj2mask = process.normalize_adj(aug_adj2mask + sp.eye(aug_adj2mask.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
    sp_aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)

    sp_aug_adj1mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj1mask)
    sp_aug_adj2mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj2mask)

else:
    adj = (adj + sp.eye(adj.shape[0])).todense()
    aug_adj1edge = (aug_adj1edge + sp.eye(aug_adj1edge.shape[0])).todense()
    aug_adj2edge = (aug_adj2edge + sp.eye(aug_adj2edge.shape[0])).todense()

    aug_adj1mask = (aug_adj1mask + sp.eye(aug_adj1mask.shape[0])).todense()
    aug_adj2mask = (aug_adj2mask + sp.eye(aug_adj2mask.shape[0])).todense()

# '''
# ------------------------------------------------------------
# mask
# ------------------------------------------------------------
'''

# '''
# ------------------------------------------------------------
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
    aug_adj1edge = torch.FloatTensor(aug_adj1edge[np.newaxis])
    aug_adj2edge = torch.FloatTensor(aug_adj2edge[np.newaxis])
    aug_adj1mask = torch.FloatTensor(aug_adj1mask[np.newaxis])
    aug_adj2mask = torch.FloatTensor(aug_adj2mask[np.newaxis])

labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
# print("labels",labels)
print("adj",sp_adj.shape)
print("feature",features.shape)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


LP = False


print("")

lista4=[0.0001]

best_accs=0
list1=[256]
list2=[0.0001]
for lr in list2:
    for hid_units in list1:
        for a4 in lista4:
            a1 = 0.72
            a2 = 0.18
            a3 = 0.1 
            model = PrePrompt(ft_size, hid_units, nonlinearity,negetive_sample,a1,a2,a3,a4,1,0.3)
            optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
            if torch.cuda.is_available():
                print('Using CUDA')
                model = model.cuda()
                # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
                features = features.cuda()
                aug_features1edge = aug_features1edge.cuda()
                aug_features2edge = aug_features2edge.cuda()
                aug_features1mask = aug_features1mask.cuda()
                aug_features2mask = aug_features2mask.cuda()
                if sparse:
                    sp_adj = sp_adj.cuda()
                    sp_aug_adj1edge = sp_aug_adj1edge.cuda()
                    sp_aug_adj2edge = sp_aug_adj2edge.cuda()
                    sp_aug_adj1mask = sp_aug_adj1mask.cuda()
                    sp_aug_adj2mask = sp_aug_adj2mask.cuda()
                else:
                    adj = adj.cuda()
                    aug_adj1edge = aug_adj1edge.cuda()
                    aug_adj2edge = aug_adj2edge.cuda()
                    aug_adj1mask = aug_adj1mask.cuda()
                    aug_adj2mask = aug_adj2mask.cuda()
                labels = labels.cuda()
                idx_train = idx_train.cuda()
                idx_val = idx_val.cuda()
                idx_test = idx_test.cuda()
            b_xent = nn.BCEWithLogitsLoss()
            xent = nn.CrossEntropyLoss()
            cnt_wait = 0
            best = 1e9
            best_t = 0
            for epoch in range(nb_epochs):
                model.train()
                optimiser.zero_grad()
                idx = np.random.permutation(nb_nodes)
                shuf_fts = features[:, idx, :]
                lbl_1 = torch.ones(batch_size, nb_nodes)
                lbl_2 = torch.zeros(batch_size, nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)
                if torch.cuda.is_available():
                    shuf_fts = shuf_fts.cuda()
                    lbl = lbl.cuda()
                loss = model(features, shuf_fts, aug_features1edge, aug_features2edge, aug_features1mask, aug_features2mask,
                            sp_adj if sparse else adj,
                            sp_aug_adj1edge if sparse else aug_adj1edge,
                            sp_aug_adj2edge if sparse else aug_adj2edge,
                            sp_aug_adj1mask if sparse else aug_adj1mask,
                            sp_aug_adj2mask if sparse else aug_adj2mask,
                            sparse, None, None, None, lbl=lbl)
                print('Loss:[{:.4f}]'.format(loss.item()))
                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), args.save_name)
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    print('Early stopping!')
                    break
                loss.backward()
                optimiser.step()
            print('Loading {}th epoch'.format(best_t))
            model.load_state_dict(torch.load(args.save_name))
            model.eval()
            embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None,LP)
            dgiprompt = model.dgi.prompt
            graphcledgeprompt = model.graphcledge.prompt
            lpprompt = model.lp.prompt
            preval_embs = embeds[0, idx_val]
            test_embs = embeds[0, idx_test]
            val_lbls = torch.argmax(labels[0, idx_val], dim=1)
            test_lbls = torch.argmax(labels[0, idx_test], dim=1)
            tot = torch.zeros(1)
            tot = tot.cuda()
            accs = []

            print('-' * 100)
            cnt_wait = 0
            best = 1e9
            best_t = 0
            for shotnum in range(1,2):
                tot = torch.zeros(1)
                tot = tot.cuda()
                accs = []
                print("shotnum",shotnum)
                for i in range(100):
                    idx_train = torch.load("data/fewshot_cora/{}-shot_cora/{}/idx.pt".format(shotnum,i)).type(torch.long).cuda()
                    pretrain_embs = embeds[0, idx_train]
                    train_lbls = torch.load("data/fewshot_cora/{}-shot_cora/{}/labels.pt".format(shotnum,i)).type(torch.long).squeeze().cuda()
                    print("true",i,train_lbls)
                    feature_prompt=featureprompt(model.dgiprompt.prompt,model.graphcledgeprompt.prompt,model.lpprompt.prompt).cuda()
                    log = downprompt(dgiprompt, graphcledgeprompt, lpprompt,a4, hid_units, nb_classes,embeds,train_lbls)
                    # opt = torch.optim.Adam(log.parameters(),downstreamprompt.parameters(),lr=0.01, weight_decay=0.0)
                    opt = torch.optim.Adam([*log.parameters(),*feature_prompt.parameters()], lr=0.001)
                    # opt = torch.optim.Adam(log.parameters(), lr=0.001)
                    log.cuda()
                    best = 1e9
                    pat_steps = 0
                    best_acc = torch.zeros(1)
                    best_acc = best_acc.cuda()
                    for _ in range(50):
                        log.train()
                        opt.zero_grad()
                        prompt_feature = feature_prompt(features)
                        # print(feature_prompt.weightprompt.weight)
                        embeds1= model.gcn(prompt_feature, sp_adj if sparse else adj, sparse, LP)
                        pretrain_embs1 = embeds1[0, idx_train]
                        logits = log(pretrain_embs,pretrain_embs1,1).float().cuda()
                        loss = xent(logits, train_lbls)
                        # print('loss' ,loss)
                        # print("predict=",torch.argmax(logits, dim=1))
                        # print("lbels",train_lbls)
                        # print("predict=",torch.argmax(logits, dim=1))
                        # print("train acc",torch.sum(torch.argmax(logits, dim=1)== train_lbls).float() / train_lbls.shape[0])
                        if loss < best:
                            best = loss
                            # best_t = epoch
                            cnt_wait = 0
                            # torch.save(model.state_dict(), args.save_name)
                        else:
                            cnt_wait += 1
                        if cnt_wait == patience:
                            print('Early stopping!')
                            break
                         
                        loss.backward(retain_graph=True)
                        opt.step()
                    prompt_feature = feature_prompt(features)
                    embeds1, _ = model.embed(prompt_feature, sp_adj if sparse else adj, sparse, None,LP)
                    test_embs1 = embeds1[0, idx_test]
                    logits = log(test_embs,test_embs1)
                    # print("logits",logits)
                    # print(log.a)
                    preds = torch.argmax(logits, dim=1)
                    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                    accs.append(acc * 100)
                    # print('acc:[{:.4f}]'.format(acc))
                    tot += acc

                print('-' * 100)
                print('Average accuracy:[{:.4f}]'.format(tot.item() / 50))
                accs = torch.stack(accs)
                print('Mean:[{:.4f}]'.format(accs.mean().item()))
                print('Std :[{:.4f}]'.format(accs.std().item()))
                print('-' * 100)
                row = [shotnum,lr,LP,hid_units,a1,a2,a3,a4,accs.mean().item(),accs.std().item()]
                out = open("data/cora_fewshot.csv", "a", newline="")
                csv_writer = csv.writer(out, dialect="excel")
                csv_writer.writerow(row)
