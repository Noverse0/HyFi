from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class ProposedConv(MessagePassing):
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, 
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False, 
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        # self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None) 
            self.register_parameter('bias_e2n', None) 
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None
    
    def forward(self, x: Tensor, hyperedge_index: Tensor, 
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None, 
                dropout_rate: Optional[float] = 0.0, noise_std: Optional[float] = 0.0):

        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)
            hypernode_weight = x.new_ones(hyperedge_index.shape[1])
            
            # noise를 생성해서 hyperedge_weight에 빼줌
            # noise = torch.abs(torch.normal(mean=0.0, std=noise_std, size=(num_edges,))).to(x.device)
            # noise = torch.normal(mean=0.0, std=noise_std, size=(num_edges,)).to(x.device)
            # hyperedge_weight = hyperedge_weight - noise
            
            # hyperedge_index만큼 noise를 만들어서 hyperedge_weight에 concat해줌
            # noise = torch.abs(torch.normal(mean=0.0, std=noise_std, size=(hyperedge_index[0].shape[0]-num_edges,))).to(x.device)
            # noise = torch.normal(mean=0.0, std=noise_std, size=(hyperedge_index[0].shape[0]-num_edges,)).to(x.device)
            # hyperedge_weight = torch.concat((hyperedge_weight, noise), dim=0)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(hypernode_weight,
                             hyperedge_index[1], dim=0, dim_size=num_edges)

            De_inv = De.pow(-1.0)
            De_inv[De_inv == float('inf')] = 0
            Dn_inv_sqrt = Dn.pow(-0.5)
            Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
            
            norm_n2e = De_inv[edge_idx] * Dn_inv_sqrt[node_idx]
            norm_e2n = Dn_inv_sqrt[node_idx]

            if self.cached:
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        x = self.lin_n2e(x)
        # e = self.propagate(hyperedge_index, x=x, norm=norm_n2e, edge_weight=hyperedge_weight,
        #                        size=(num_nodes, num_edges))  # Node to edge
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e,
                               size=(num_nodes, num_edges))  # Node to edge

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=dropout_rate, training=self.training)
        
        x = self.lin_e2n(e)
        # n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n, edge_weight=hyperedge_weight,
        #                        size=(num_edges, num_nodes))  # Edge to node
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n,
                               size=(num_edges, num_nodes))  # Edge to node

        if self.bias_e2n is not None:
            n = n + self.bias_e2n
        n = self.act(n)
        return n, e # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j
    