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
from .crossvit import CrossViT


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


class SGMF(torch.nn.Module):
    def __init__(self, input_dim=2227, num_layers=4, edge_agg='spatial', multires=False, resample=0,
                 fusion=None, num_features=1024, hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=4):
        super(SGMF, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample
        self.feature_dimensions = 1024  # dino:384, simclr:512, ImageNet:1024

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(self.feature_dimensions, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(self.feature_dimensions, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers_s = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv_s = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer_s = DeepGCNLayer(conv_s, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers_s.append(layer_s)
        self.path_phi_s = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])  


        self.layers_l = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv_l = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer_l = DeepGCNLayer(conv_l, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers_l.append(layer_l)
        self.path_phi_l = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])

        # define tissue-graph gcn
        self.fc_tissue = nn.Sequential(*[nn.Linear(self.feature_dimensions, 512), nn.ReLU()])
        self.tissue_graph = Tissue_GCN_cls().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.coattn_s = MultiheadAttention(embed_dim=hidden_dim*4, num_heads=1)
        self.coattn_l = MultiheadAttention(embed_dim=hidden_dim*4, num_heads=1)

        # cross attention
        self.crossattn = CrossViT(num_classes=3)

        # gated_attention
        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*4, D=hidden_dim*4, n_classes=1)

        self.classifier = torch.nn.Linear(hidden_dim*4, n_classes)

    def forward(self, data_s, data_l, tissue_data, attn_mask_s, attn_mask_l, attention_only=False):

        if self.edge_agg == 'spatial':
            edge_index_s = data_s.edge_index
            edge_index_l = data_l.edge_index
        elif self.edge_agg == 'latent':
            edge_index_s = data_s.edge_latent
            edge_index_l = data_l.edge_indel

        batch_s = data_s.batch
        batch_l = data_l.batch
        edge_attr_s = None
        edge_attr_l = None
        centroid = tissue_data['centroid'] / 10.  # 缩小十倍
        centroid = centroid.round()

        # gcn for small scale
        x_s = self.fc(data_s.x)
        x1_s = x_s
        x_s = self.layers_s[0].conv(x1_s, edge_index_s, edge_attr_s)
        x1_s = torch.cat([x1_s, x_s], axis=1)
        for layer_s in self.layers_s[1:]:
            x_s = layer_s(x_s, edge_index_s, edge_attr_s)
            x1_s = torch.cat([x1_s, x_s], axis=1)  # concate every layer's output together
        h_path_s = x1_s
        h_path_s = self.path_phi_s(h_path_s).unsqueeze(0)  # each node's feature

        # gcn for large scale
        x_l = self.fc(data_l.x)
        x1_l = x_l
        x_l = self.layers_l[0].conv(x1_l, edge_index_l, edge_attr_l)
        x1_l = torch.cat([x1_l, x_l], axis=1)
        for layer_l in self.layers_l[1:]:
            x_l = layer_l(x_l, edge_index_l, edge_attr_l)
            x1_l = torch.cat([x1_l, x_l], axis=1)  # concate every layer's output together
        h_path_l = x1_l
        h_path_l = self.path_phi_l(h_path_l).unsqueeze(0)  # each node's feature

        # tissue features
        tissue_feature = self.fc_tissue(tissue_data.x)
        tissue_feature = tissue_feature.unsqueeze(1)

        # self_attention for small and large patches
        h_path_s = torch.cat((tissue_feature.permute(1,0,2), h_path_s), dim=1)
        tmp_mask_s = torch.eye(tissue_feature.shape[0]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
        tmp_mask_s = ~ tmp_mask_s.type(torch.bool)
        attn_mask_s_new = torch.cat((tmp_mask_s, attn_mask_s), dim=-1)

        h_path_l = torch.cat((tissue_feature.permute(1,0,2), h_path_l), dim=1)
        tmp_mask_l = torch.eye(tissue_feature.shape[0]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
        tmp_mask_l = ~ tmp_mask_l.type(torch.bool)
        attn_mask_l_new = torch.cat((tmp_mask_l, attn_mask_l), dim=-1)

        h_path_cooatn_s, A_cooatn_s = self.coattn_s(tissue_feature, h_path_s.squeeze(), h_path_s.squeeze(), attn_mask=attn_mask_s_new.squeeze())  # attn_mask
        h_path_cooatn_l, A_cooatn_l = self.coattn_l(tissue_feature, h_path_l.squeeze(), h_path_l.squeeze(), attn_mask=attn_mask_l_new.squeeze())

        h_fusion = torch.cat((h_path_cooatn_s, h_path_cooatn_l), dim=-1).squeeze(0)
        tissue_data.x = h_fusion.squeeze()
        h_fusion, tissue_edge_index = self.tissue_graph(tissue_data)


        A_fusion, h_fusion = self.path_attention_head(h_fusion)
        A_fusion = torch.transpose(A_fusion, 1, 0)
        h_fusion = torch.mm(F.softmax(A_fusion, dim=1), h_fusion).squeeze()

        logits = self.classifier(h_fusion).unsqueeze(0)
        save_logits = logits

        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]

        return logits, Y_prob, Y_hat, save_logits


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
        x_ = x
        x = self.layers[0].conv(x, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        x = x_

        return x, edge_index