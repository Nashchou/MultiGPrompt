import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import copy
from functools import partial
from dgl.nn.pytorch.conv import RelGraphConv
# from basemodel import GraphAdjModel
import numpy as np
import tqdm

class Lp(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lp, self).__init__()
        self.sigm = nn.ELU()
        self.act=torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()



    def forward(self,gcn,seq,adj,sparse):
        h_1 = gcn(seq,adj,sparse,True)
        # ret = h_1
        ret = h_1 * self.prompt
        # print("ret1",ret)
        ret = self.sigm(ret.squeeze(dim=0))
                # print("ret2",ret)
        # ret = ret.squeeze(dim=0)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
