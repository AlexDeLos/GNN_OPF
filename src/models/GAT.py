import torch as th
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):

    class_name = "GAT"

    def __init__(
            self, 
            input_dim, 
            output_dim, 
            edge_attr_dim,
            n_hidden_conv=8, 
            hidden_conv_dim=64, 
            n_hidden_lin=4, 
            hidden_lin_dim=64, 
            dropout_rate=0.1, 
            heads=1,
            *args, 
            **kwargs
        ):
        
        super().__init__(*args, **kwargs)
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()

        self.convs.append(GATConv(in_channels=input_dim, out_channels=hidden_conv_dim, heads=heads, edge_attr_dim=edge_attr_dim))
        for _ in range(n_hidden_conv):
            self.convs.append(GATConv(in_channels=hidden_conv_dim * heads, out_channels=hidden_conv_dim, heads=heads, edge_attr_dim=edge_attr_dim))

        self.lins.append(nn.Linear(hidden_conv_dim * heads, hidden_lin_dim))
        for _ in range(n_hidden_lin):
            self.lins.append(nn.Linear(hidden_lin_dim, hidden_lin_dim))

        self.lins.append(nn.Linear(hidden_lin_dim, output_dim))
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for c in self.convs:
            x = c(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.leaky_relu(x, 0.2)
                
        for l in self.lins[:-1]:
            x = l(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.relu(x, 0.2)
        x = self.lins[-1](x)

        return x