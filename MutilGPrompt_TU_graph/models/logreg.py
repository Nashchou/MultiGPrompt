import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN
import scipy.sparse as sp
from utils import process

class LogReg(nn.Module):
    def __init__(self,ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.nb_classes = nb_classes
        self.hid_units = ft_in
        self.act = nn.PReLU()
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self,seq,graph_len):
        pretrain_embs = split_and_batchify_graph_feats(seq, graph_len)
        # pretrain_embs = self.act(pretrain_embs)
        ret = self.fc(pretrain_embs)
        return self.act(ret)

def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):

    # print("graphsize",graph_sizes)
    # print("sum",torch.sum(graph_sizes))
    cnt = 0 

    result = torch.FloatTensor(graph_sizes.shape[0], batched_graph_feats.shape[1]).cuda()

    for i in range(graph_sizes.shape[0]):
        # print("i",i)
        current_graphlen = int(graph_sizes[i].item())
        graphlen = range(cnt,cnt+current_graphlen)
        # print("graphlen",graphlen)
        result[i] = torch.sum(batched_graph_feats[graphlen], dim=0)
        cnt = cnt + current_graphlen
    # print("resultsum",cnt)    
    return result