from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import dgl.sparse as dglsp
import numpy as np
from model.layers import ProposedConv

class HGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, act: Callable = nn.PReLU()):
        super(HGNN, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False))
        else:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False))
            for _ in range(self.num_layers - 2):
                self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False))
            self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            
    def add_weight_noise(self, noise_std: float):
        # weight에 가우시안 노이즈 추가
        for conv in self.convs:
            # gaussian noise
            conv.lin_n2e.weight.data += conv.lin_n2e.weight.data * torch.normal(0, noise_std, size=conv.lin_n2e.weight.size()).to(conv.lin_n2e.weight.device)
            conv.lin_e2n.weight.data += conv.lin_e2n.weight.data * torch.normal(0, noise_std, size=conv.lin_e2n.weight.size()).to(conv.lin_e2n.weight.device)

            # uniform noise
            # conv.lin_n2e.weight.data += conv.lin_n2e.weight.data * torch.rand(size=conv.lin_n2e.weight.size()).to(conv.lin_n2e.weight.device) * noise_std
            # conv.lin_e2n.weight.data += conv.lin_e2n.weight.data * torch.rand(size=conv.lin_e2n.weight.size()).to(conv.lin_e2n.weight.device) * noise_std
            
            # poisson noise
            # conv.lin_n2e.weight.data += conv.lin_n2e.weight.data * torch.poisson(torch.full(conv.lin_n2e.weight.size(), noise_std)).to(conv.lin_n2e.weight.device)
            # conv.lin_e2n.weight.data += conv.lin_e2n.weight.data * torch.poisson(torch.full(conv.lin_e2n.weight.size(), noise_std)).to(conv.lin_e2n.weight.device)
    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int, dropout_rate: float, noise_std: float):
        for i in range(self.num_layers):
            if noise_std != 0.0:
                self.add_weight_noise(noise_std)
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges, dropout_rate, noise_std)
            x = self.act(x)
        return x, e # act, act
    
class FG_HGCL(nn.Module):
    def __init__(self, encoder: HGNN, proj_dim: int, device):
        super(FG_HGCL, self).__init__()
        self.encoder = encoder
        self.device = device

        self.node_dim = self.encoder.node_dim
        self.edge_dim = self.encoder.edge_dim

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)

        self.reset_parameters()
        
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        
    def forward(self, x: Tensor, hyperedge_index: Tensor, 
                dropout_rate: Optional[float] = 0.0, noise_std_n: Optional[float] = 0.0, noise_std_e: Optional[float] = 0.0, aug_edge_num: Optional[int] = 0,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
        
        if noise_std_e != 0.0:
            noise_x = self.add_gaussian_noise(x, noise_std_n)
            n, e = self.encoder(noise_x, self_loop_hyperedge_index, num_nodes, num_edges + num_nodes, dropout_rate, noise_std_e)
        else:
            n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + num_nodes, dropout_rate, noise_std_e)
        return n, e[:num_edges]
        
    def add_gaussian_noise(self, feature, std=0.1):
        noise = torch.abs(torch.normal(0, std, size=feature.size())).to(self.device)
        noise_feature = torch.where(feature == 1, feature - noise, feature + noise)
        
        # uniform noise
        # noise = torch.rand(size=feature.size()).to(self.device) * std
        # noise_feature = torch.where(feature == 1, feature - noise, feature + noise)
        
        # bernoulli noise
        # noise = torch.bernoulli(torch.full(feature.size(), std)).to(self.device)
        # noise_feature = torch.where(feature == 1, feature - noise, feature + noise)
        
        # poisson noise
        # noise = torch.poisson(torch.full(feature.size(), std)).to(self.device)
        # noise_feature = feature + noise
        
        return noise_feature

    def f(self, x, tau):
        return torch.exp(x / tau)
    
    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))
    
    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))
    
    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calulate_loss(self, n1: Tensor, n2: Tensor,  overlab_hyperedge: Tensor, 
                    tau: float,
                    batch_size: Optional[int] = None,
                    mean: bool = True):

        if batch_size != None:
            loss = self.semi_loss_batch(n1, n2, tau, overlab_hyperedge, batch_size)
        else:
            loss = self.semi_loss(n1, n2, tau, overlab_hyperedge)
            # loss += self.semi_loss_batch(n2, n1, tau, overlab_hyperedge, w_wp, w_wn, batch_size, detail)
            
        loss = loss.mean() if mean else loss.sum()
        
        return loss

    def semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, overlab_hyperedge: Tensor, 
                    batch_size: int):
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=self.device)
        weight_hyperedge = (overlab_hyperedge / overlab_hyperedge.diag())
        losses = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            current_h1 = h1[start_idx:end_idx]
            # current_h2 = h2[start_idx:end_idx]
            current_overlab_hyperedge = overlab_hyperedge[start_idx:end_idx, start_idx:end_idx]
            current_weight_hyperedge = weight_hyperedge[start_idx:end_idx, start_idx:end_idx]

            self_sim = self.f(self.cosine_similarity(current_h1, current_h1), tau)
            # between_sim = self.f(self.cosine_similarity(current_h1, current_h2), tau)
            diag_mask = torch.eye(end_idx - start_idx, device=self.device, dtype=torch.bool) 
            
            pos_sim = 0
            for i in range(h2.shape[0]):
                pos_sim += (self.f(self.cosine_similarity(current_h1, h2[i, start_idx:end_idx]), tau) * diag_mask).sum(1)
            weak_pos_sim = (self_sim * (current_overlab_hyperedge > 0 & ~diag_mask) * current_overlab_hyperedge * current_weight_hyperedge).sum(1)
            
            # no weak weight
            # weak_pos_sim = (self_sim * (current_overlab_hyperedge > 0 & ~diag_mask) * current_overlab_hyperedge).sum(1)
            
            # no group contrastive
            # weak_pos_sim = (self_sim * (current_overlab_hyperedge > 0 & ~diag_mask) * (current_overlab_hyperedge / current_overlab_hyperedge.diag())).sum(1)
            
            # no weak weight & no group contrastive
            # weak_pos_sim = (self_sim * (current_overlab_hyperedge > 0 & ~diag_mask)).sum(1)
            
            neg_sim = (self_sim * (current_overlab_hyperedge == 0)).sum(1)
            
            # loss = -torch.log((pos_sim) / (pos_sim + neg_sim))
            loss = -torch.log((pos_sim + weak_pos_sim) / (pos_sim + weak_pos_sim + neg_sim))
            
            losses.append(loss)
        return torch.cat(losses)

    
    def semi_loss(self, n1: torch.Tensor, n2: torch.Tensor, tau: float, overlab_hyperedge: Tensor):
        diag_mask = torch.eye(overlab_hyperedge.size(0), dtype=torch.bool, device=self.device)
        pos_sim = 0
        for i in range(n2.shape[0]):
            pos_sim += (self.f(self.cosine_similarity(n1, n2[i]), tau) * diag_mask).sum(1)
        
        self_sim = self.f(self.cosine_similarity(n1, n1), tau)
        weak_pos_sim = (self_sim * (overlab_hyperedge > 0 & ~diag_mask) * overlab_hyperedge * (overlab_hyperedge / overlab_hyperedge.diag())).sum(1)
        
        # no weak weight
        # weak_pos_sim = (self_sim * (overlab_hyperedge > 0 & ~diag_mask) * overlab_hyperedge).sum(1)
        
        # no group contrastive
        # weak_pos_sim = (self_sim * (overlab_hyperedge > 0 & ~diag_mask) * (overlab_hyperedge / overlab_hyperedge.diag())).sum(1)
        
        # no weak weight & no group contrastive
        # weak_pos_sim = (self_sim * (overlab_hyperedge > 0 & ~diag_mask)).sum(1)
        
        neg_sim = (self_sim * (overlab_hyperedge == 0)).sum(1)
        
        # return -torch.log((pos_sim) / (pos_sim + neg_sim))
        return -torch.log((pos_sim + weak_pos_sim) / (pos_sim + weak_pos_sim + neg_sim))
