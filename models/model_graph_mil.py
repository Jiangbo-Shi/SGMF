from os.path import join
from collections import OrderedDict

import pdb
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.transforms.normalize_features import NormalizeFeatures

from models.model_utils import *


class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class DeepGraphConv_Surv(torch.nn.Module):
    def __init__(self, edge_agg='latent', resample=0, num_features=1024, hidden_dim=256, 
        linear_dim=256, use_edges=False, dropout=0.25, n_classes=4):
        super(DeepGraphConv_Surv, self).__init__()
        self.use_edges = use_edges
        self.resample = resample
        self.edge_agg = edge_agg

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample)])

        self.conv1 = GINConv(Seq(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        self.path_attention_head = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)

        self.node_classifier = torch.nn.Linear(hidden_dim, 8)

        self.pool = False

    def relocate(self):
        from torch_geometric.nn import DataParallel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.conv1 = nn.DataParallel(self.conv1, device_ids=device_ids).to('cuda:0')
            self.conv2 = nn.DataParallel(self.conv2, device_ids=device_ids).to('cuda:0')
            self.conv3 = nn.DataParallel(self.conv3, device_ids=device_ids).to('cuda:0')
            self.path_attention_head = nn.DataParallel(self.path_attention_head, device_ids=device_ids).to('cuda:0')

        self.path_rho = self.path_rho.to(device)
        self.classifier = self.classifier.to(device)
        self.node_classifier = self.node_classifier.to(device)


    def forward(self, data, attention_only=False):
        # data = data['x_path']
        x = data.x
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        batch = data.batch
        edge_attr = None

        if self.resample:
            x = self.fc(x)

        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        if self.pool:
            x1, edge_index, _, batch, perm, score = self.pool1(x1, edge_index, None, batch)
            x1_cat = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        if self.pool:
            x2, edge_index, _, batch, perm, score = self.pool2(x2, edge_index, None, batch)
            x2_cat = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)

        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))

        h_path = x3

        Y_node = self.node_classifier(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)

        A_raw = A_path
        if(attention_only):
            return A_raw

        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path = self.path_rho(h_path).squeeze()
        h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        # return hazards, S, Y_hat, A_path, None
        return logits, Y_prob, Y_hat, A_raw, S, Y_node


class Patch_GCN_cls(torch.nn.Module):
    def __init__(self, input_dim=2227, num_layers=4, edge_agg='spatial', multires=False, resample=0,
        fusion=None, num_features=1024, hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=4):
        super(Patch_GCN_cls, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])  # Dropout：随机将输入张量中的部分元素置为零

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*4, D=hidden_dim*4, n_classes=1)

        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU()], nn.Dropout(dropout))

        self.classifier = torch.nn.Linear(hidden_dim*8, n_classes)

        # define tissue-graph gcn
        self.tissue_graph = Tissue_GCN_cls().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    def forward(self, data, tissue_data, attention_only=False):

        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        batch = data.batch
        edge_attr = None

        x = self.fc(data.x)
        x_ = x

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)  # concate every layer's output together

        h_path = x_
        h_path = self.path_phi(h_path)  # each node's feature

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        A_raw = A_path
        if(attention_only):
            return A_raw

        h_path_attn = torch.mm(F.softmax(A_path, dim=1), h_path)  # attention-based feature fusion
        h_path_attn = self.path_rho(h_path_attn).squeeze()

        # mean pooling
        # h = torch.mean(h_path, axis=0)

        tissue_feature = self.tissue_graph(tissue_data).squeeze()
        tissue_feature = torch.mean(tissue_feature, axis=0)


        combine_feature = torch.cat((h_path_attn, tissue_feature), 0)

        # A_path, tissue_feature = self.path_attention_head(tissue_feature)
        # A_path = torch.transpose(A_path, 1, 0)
        #
        # A_raw = A_path
        # if(attention_only):
        #     return A_raw
        #
        # tissue_feature = torch.mm(F.softmax(A_path, dim=1), tissue_feature)  # attention-based feature fusion

        # h = self.path_rho(combine_feature).squeeze()

        logits = self.classifier(combine_feature).unsqueeze(0)  # logits needs to be a [1 x 4] vector

        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]

        return logits, Y_prob, Y_hat, A_raw


# class Patch_GCN_cls(torch.nn.Module):
#     def __init__(self, input_dim=2227, num_layers=4, edge_agg='spatial', multires=False, resample=0,
#                  fusion=None, num_features=1024, hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=4):
#         super(Patch_GCN_cls, self).__init__()
#         self.use_edges = use_edges
#         self.fusion = fusion
#         self.pool = pool
#         self.edge_agg = edge_agg
#         self.multires = multires
#         self.num_layers = num_layers-1
#         self.resample = resample
#
#         if self.resample > 0:
#             self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.25)])
#         else:
#             self.fc = nn.Sequential(*[nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.25)])
#
#         self.layers = torch.nn.ModuleList()
#         for i in range(1, self.num_layers+1):
#             conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
#                            t=1.0, learn_t=True, num_layers=2, norm='layer')
#             norm = LayerNorm(hidden_dim, elementwise_affine=True)
#             act = ReLU(inplace=True)
#             layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
#             self.layers.append(layer)
#
#         self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])  # Dropout：随机将输入张量中的部分元素置为零
#
#         self.path_attention_head = Attn_Net_Gated(L=hidden_dim*4, D=hidden_dim*4, n_classes=1)
#
#         self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU()], nn.Dropout(dropout))
#
#         self.classifier = torch.nn.Linear(hidden_dim*12, n_classes)
#         # self.node_classifier = torch.nn.Linear(hidden_dim*4, 8)
#
#         # define tissue-graph gcn
#         self.tissue_graph = Tissue_GCN_cls().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#
#         self.coattn = MultiheadAttention(embed_dim=hidden_dim*4, num_heads=1)
#
#     def forward(self, data, tissue_data, attention_only=False):
#
#         if self.edge_agg == 'spatial':
#             edge_index = data.edge_index
#         elif self.edge_agg == 'latent':
#             edge_index = data.edge_latent
#
#         batch = data.batch
#         edge_attr = None
#
#         x = self.fc(data.x)
#         x_ = x
#
#         x = self.layers[0].conv(x_, edge_index, edge_attr)
#         x_ = torch.cat([x_, x], axis=1)
#         for layer in self.layers[1:]:
#             x = layer(x, edge_index, edge_attr)
#             x_ = torch.cat([x_, x], axis=1)  # concate every layer's output together
#
#         h_path = x_
#         h_path = self.path_phi(h_path)  # each node's feature
#
#         A_path, h_path = self.path_attention_head(h_path)
#         A_path = torch.transpose(A_path, 1, 0)
#         A_raw = A_path
#         if(attention_only):
#             return A_raw
#
#         h_path_attn = torch.mm(F.softmax(A_path, dim=1), h_path)  # attention-based feature fusion
#
#         h_path_attn = self.path_rho(h_path_attn).squeeze()
#
#         # mean pooling
#         # h = torch.mean(h_path, axis=0)
#
#         tissue_feature = self.tissue_graph(tissue_data).squeeze()
#         tissue_feature = torch.mean(tissue_feature, axis=0)
#
#         # self_attention
#         h_path_cooatn, A_cooatn = self.coattn(tissue_feature.unsqueeze(0).unsqueeze(0), h_path.unsqueeze(0), h_path.unsqueeze(0))
#
#         combine_feature = torch.cat((h_path_attn, tissue_feature, h_path_cooatn.squeeze()), 0)
#
#         # A_path, tissue_feature = self.path_attention_head(tissue_feature)
#         # A_path = torch.transpose(A_path, 1, 0)
#         #
#         # A_raw = A_path
#         # if(attention_only):
#         #     return A_raw
#         #
#         # tissue_feature = torch.mm(F.softmax(A_path, dim=1), tissue_feature)  # attention-based feature fusion
#
#         # h = self.path_rho(combine_feature).squeeze()
#
#         logits = self.classifier(combine_feature).unsqueeze(0)  # logits needs to be a [1 x 4] vector
#
#         Y_prob = F.softmax(logits, dim = 1)
#         Y_hat = torch.topk(logits, 1, dim = 1)[1]
#
#         return logits, Y_prob, Y_hat, Y_hat


class Tissue_GCN_cls(torch.nn.Module):
    def __init__(self, edge_agg='spatial', hidden_dim=128):
        super(Tissue_GCN_cls, self).__init__()
        self.edge_agg = edge_agg
        self.num_layers = 3

        self.fc = nn.Sequential(*[nn.Linear(1024, 128), nn.ReLU()])

        # define tissue-graph gcn
        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', ckpt_grad=i % 3)
            self.layers.append(layer)


    def forward(self, data):

        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        edge_index = edge_index.long()
        edge_attr = None
        batch = data.batch

        x = self.fc(data.x)

        x = self.layers[0].conv(x, edge_index, edge_attr)
        x_ = torch.cat([x, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        # x = gavgp(x_, batch)
        x = x_

        return x