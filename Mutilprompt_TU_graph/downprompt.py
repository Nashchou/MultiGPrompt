import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL
from layers import GCN, AvgReadout


class downprompt(nn.Module):
    def __init__(self, prompt1, prompt2, prompt3, ft_in, nb_classes):
        super(downprompt, self).__init__()
        # self.prompt1 = prompt1
        # self.prompt2 = prompt2
        # self.prompt3 = prompt3
        self.downprompt = downstreamprompt(ft_in)
        # self.dropout = nn.Dropout(p=0.0)
        self.nb_classes = nb_classes

        self.leakyrelu = nn.ELU()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        # self.prompt = prompt3
        # self.prompt = prompt3
        # self.a = nn.Parameter(torch.FloatTensor(1, 3), requires_grad=True).cuda()
        # self.reset_parameters()
        self.nodelabelprompt = weighted_prompt(3)

        self.dffprompt = weighted_feature(2)

        self.aveemb0 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb1 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb2 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb3 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb4 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb5 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb6 = torch.FloatTensor(ft_in, ).cuda()

        self.one = torch.ones(1, ft_in).cuda()

        # for x in range(0, nb_classes):
        #     if labels[x].item() == 0:
        #         self.aveemb0 = feature[index[x].item()]
        #     if labels[x].item() == 1:
        #         self.aveemb1 = feature[index[x].item()]
        #     if labels[x].item() == 2:
        #         self.aveemb2 = feature[index[x].item()]
        #     if labels[x].item() == 3:
        #         self.aveemb3 = feature[index[x].item()]
        #     if labels[x].item() == 4:
        #         self.aveemb4 = feature[index[x].item()]
        #     if labels[x].item() == 5:
        #         self.aveemb5 = feature[index[x].item()]
        #     if labels[x].item() == 6:
        #         self.aveemb6 = feature[index[x].item()]

        self.ave = torch.FloatTensor(nb_classes, ft_in).cuda()
        # print("avesize",self.ave.size(),"ave",self.ave)

        # emb0 = torch.zeros(1, 1, 512).cuda()
        # emb1 = torch.zeros(1, 1, 512).cuda()
        # emb2 = torch.zeros(1, 1, 512).cuda()
        # emb3 = torch.zeros(1, 1, 512).cuda()
        # emb4 = torch.zeros(1, 1, 512).cuda()
        # emb5 = torch.zeros(1, 1, 512).cuda()
        # emb6 = torch.zeros(1, 1, 512).cuda()

        # for x in range(0, embs.shape[0]):
        #     if lbls[x] == 0:
        #         emb0 =torch.mean(torch.stack((emb0.squeeze(0), embs[x].unsqueeze(0)), dim=1))
        #     if lbls[x] == 1:
        #         emb1 = torch.mean(torch.stack((emb0.squeeze(0), embs[x].unsqueeze(0)), dim=1))
        #     if lbls[x] == 2:
        #         emb2 = torch.mean(torch.stack((emb0.squeeze(0), embs[x].unsqueeze(0)), dim=1))
        #     if lbls[x] == 3:
        #         print('emb3', emb3.squeeze(0).shape)
        #         print('ems[3]', embs[x].unsqueeze(0).shape)
        #         emb3 = torch.mean(torch.stack((emb0.squeeze(0), embs[x].unsqueeze(0)), dim=1))
        #     if lbls[x] == 4:
        #         print('ems[4]', embs[x].squeeze(0).unsqueeze(0).shape)
        #         emb4 = torch.stack((emb4.squeeze(0), embs[x].unsqueeze(0)), dim=1)
        #     if lbls[x] == 5:
        #         emb5 = torch.stack((emb5.squeeze(0), embs[x].unsqueeze(0)), dim=1)
        #     if lbls[x] == 6:
        #         emb6 = torch.stack((emb6.squeeze(0), embs[x].unsqueeze(0)), dim=1)

    def forward(self, seq,graph_len):
        # promptweight = torch.FloatTensor(1,3).cuda()
        # promptweight[0][0] = 0.3
        # promptweight[0][1] = 0.3
        # promptweight[0][2] = 0.3
        # # print(self.a)
        weight = self.nodelabelprompt(self.prompt)
        weight = self.one + weight
        # # # weight = torch.mm(promptweight,self.prompt)
        # # # # print("weight",self.a.weight)
        rawret1 = weight * seq
        
        rawret2 = self.downprompt(seq)

        rawret = self.dffprompt(rawret1 ,rawret2)

        # rawret = seq
        rawret = rawret.cuda()
        
        graph_embedding=split_and_batchify_graph_feats(rawret, graph_len)
        # print(graph_embedding.shape)
        # out = self.dropout(graph_embedding)

        
        return graph_embedding
        # # rawret = torch.stack((rawret,rawret,rawret,rawret,rawret,rawret))
        # if train == 1:
        #     self.ave = averageemb(labels=self.labels, rawret=rawret, nb_class=self.nb_classes)
        #     # if self.labels[x].item() == 6:
        #     #     self.aveemb6 = rawret[x]
        # # self.ave = weight * self.ave
        # # print("rawretsize",rawret.size())



        # ret = torch.FloatTensor(seq.shape[0], self.nb_classes).cuda()
        # # print("avesize",self.ave.size(),"ave",self.ave)
        # # print("rawret=", rawret[1])
        # # print("aveemb", self.ave)
        # for x in range(0, seq.shape[0]):
        #     ret[x][0] = torch.cosine_similarity(rawret[x], self.ave[0], dim=0)
        #     ret[x][1] = torch.cosine_similarity(rawret[x], self.ave[1], dim=0)
        #     if self.nb_classes == 6:
        #         ret[x][2] = torch.cosine_similarity(rawret[x], self.ave[2], dim=0)
        #         ret[x][3] = torch.cosine_similarity(rawret[x], self.ave[3], dim=0)
        #         ret[x][4] = torch.cosine_similarity(rawret[x], self.ave[4], dim=0)
        #         ret[x][5] = torch.cosine_similarity(rawret[x], self.ave[5], dim=0)
        #         # ret[x][6] = torch.cosine_similarity(rawret[x], self.ave[6], dim=0)

        # ret = F.softmax(ret, dim=1)

        # # ret = torch.argmax(ret, dim=1)
        # # print('ret=', ret)

        # return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)



def predict(graphnum,nb_classes,rawret,ave):
    # print(graphnum)
    ret = torch.FloatTensor(graphnum, nb_classes).cuda()
    # print("avesize",ave.shape,"ave",ave)
    # print("rawret=", rawret[1])
    # print("aveemb", self.ave)
    for x in range(0, graphnum):
        ret[x][0] = torch.cosine_similarity(rawret[x], ave[0], dim=0)
        ret[x][1] = torch.cosine_similarity(rawret[x], ave[1], dim=0)
        if nb_classes == 6:
            ret[x][2] = torch.cosine_similarity(rawret[x], ave[2], dim=0)
            ret[x][3] = torch.cosine_similarity(rawret[x], ave[3], dim=0)
            ret[x][4] = torch.cosine_similarity(rawret[x], ave[4], dim=0)
            ret[x][5] = torch.cosine_similarity(rawret[x], ave[5], dim=0)
    # print("retshape1",ret.shape)
    
    
    ret = F.log_softmax(ret, dim=1)
    # print("retshape2",ret.shape)
    return ret


def averageemb(labels, rawret, nb_class):
    retlabel = torch.FloatTensor(nb_class, int(rawret.shape[0] / nb_class), int(rawret.shape[1])).cuda()
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    cnt7 = 0
    # print("labels",retlabel.shape)
    # print("rawret",rawret.shape)
    for x in range(0, rawret.shape[0]):
        if labels[x].item() == 0:
            retlabel[0][cnt1] = rawret[x]
            cnt1 = cnt1 + 1
        if labels[x].item() == 1:
            retlabel[1][cnt2] = rawret[x]
            cnt2 = cnt2 + 1
        if labels[x].item() == 2:
            retlabel[2][cnt3] = rawret[x]
            cnt3 = cnt3 + 1
        if labels[x].item() == 3:
            retlabel[3][cnt4] = rawret[x]
            cnt4 = cnt4 + 1
        if labels[x].item() == 4:
            retlabel[4][cnt5] = rawret[x]
            cnt5 = cnt5 + 1
        if labels[x].item() == 5:
            retlabel[5][cnt6] = rawret[x]
            cnt6 = cnt6 + 1
        if labels[x].item() == 6:
            retlabel[6][cnt7] = rawret[x]
            cnt7 = cnt7 + 1
    retlabel = torch.mean(retlabel, dim=1)
    return retlabel



def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):


    # print("nodeemb",batched_graph_feats)
    # print(graph_sizes)
    # print("graphsize",graph_sizes)
    # print("sum",torch.sum(graph_sizes))
    cnt = 0 

    result = torch.FloatTensor(graph_sizes.shape[0], batched_graph_feats.shape[1]).cuda()

    for i in range(graph_sizes.shape[0]):
        # print("i",i)
        current_graphlen = int(graph_sizes[i].item())
        graphlen = range(cnt,cnt+current_graphlen)
        # print("graphlen",graphlen)
        result[i] = torch.sum(batched_graph_feats[graphlen,:], dim=0)
        cnt = cnt + current_graphlen
    # print("resultsum",cnt)    
    return result


# def center_embedding(input, index, label_num=7, debug=False):
#     result = torch.zeros(7, 512).cuda()
#     device = input.device
#     index = index.to(device)
#     mean = torch.ones(index.size(0)).to(device)
#     _mean = torch.zeros(label_num, device=device).scatter_add_(dim=0, index=index, src=mean).to(device)
#     index = index.reshape(-1, 1)
#     index = index.expand(input.size())
#     print("label_num", label_num)
#     print("inputsize", input.size(1))
#     result = result.scatter_add_(dim=0, index=index, src=input)
#     _mean = _mean.reshape(-1, 1)
#     result = result / _mean
#     return result
#
#
# def distance2center(f, center):
#     _f = torch.broadcast_to(f, (center.size(0), f.size(0), f.size(1)))
#     _center = torch.broadcast_to(center, (f.size(0), center.size(0), center.size(1)))
#     _f = _f.permute(1, 0, 2)
#     _center = _center.reshape(-1, _center.size(2))
#     _f = _f.reshape(-1, _f.size(2))
#     cos = torch.cosine_similarity(_f, _center, dim=1)
#     res = cos
#     res = res.reshape(f.size(0), center.size(0))
#     return res

class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0.9)
        self.weight[0][1].data.fill_(0.9)
        self.weight[0][2].data.fill_(0.1)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        # weight = F.softmax(self.weight, dim=1)
        # print("weight",weight)
        graph_embedding = torch.mm(self.weight, graph_embedding)
        return graph_embedding


class weighted_feature(nn.Module):
    def __init__(self, weightednum):
        super(weighted_feature, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(1)
        self.weight[0][1].data.fill_(0)

    def forward(self, graph_embedding1, graph_embedding2):
        
        # weight = F.softmax(self.weight, dim=1)
        # print("weight",weight)
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)


class downstreamprompt(nn.Module):
    def __init__(self, hid_units):
        super(downstreamprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.reset_parameters()
        self.act = nn.ELU()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = self.weight * graph_embedding
        return graph_embedding
    


def distance2center(input,center):
    n = input.size(0)
    m = input.size(1)
    k = center.size(0)
    input_power = torch.sum(input * input, dim=1, keepdim=True).expand(n, k)
    center_power = torch.sum(center * center, dim=1).expand(n, k)
    temp1=input_power+center_power
    temp2=2*torch.mm(input,center.transpose(0,1))
    distance = input_power + center_power - 2 * torch.mm(input, center.transpose(0, 1))
    return distance



def onehot(label, nb_classes):
    ret = torch.zeros(label.shape[0], nb_classes).cuda()
    for x in range(0, label.shape[0]):
        ret[x][int(label[x].item())] = 1
    return ret