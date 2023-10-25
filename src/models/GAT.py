import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn.models import JumpingKnowledge

class GAT(nn.Module):

    class_name = "GAT"

    def __init__(
            self, 
            input_dim, 
            output_dim, 
            edge_attr_dim,
            n_hidden_conv=2, 
            hidden_conv_dim=32, 
            n_hidden_lin=2, 
            hidden_lin_dim=16, 
            dropout_rate=0.1, 
            heads=8,
            jumping_knowledge=False,
            no_lin=False,
            *args, 
            **kwargs
        ):
        
        super().__init__(*args, **kwargs)
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.no_lin = no_lin

        self.convs.append(GATv2Conv(in_channels=input_dim, out_channels=hidden_conv_dim, heads=heads, edge_dim=edge_attr_dim))
        for _ in range(n_hidden_conv):
            self.convs.append(GATv2Conv(in_channels=hidden_conv_dim * heads, out_channels=hidden_conv_dim, heads=heads, edge_dim=edge_attr_dim))

        if self.no_lin:
            self.convs.append(GATv2Conv(hidden_conv_dim, output_dim, edge_dim=edge_attr_dim))

        self.lins.append(nn.Linear(hidden_conv_dim * heads, hidden_lin_dim))
        for _ in range(n_hidden_lin):
            self.lins.append(nn.Linear(hidden_lin_dim, hidden_lin_dim))

        self.lins.append(nn.Linear(hidden_lin_dim, output_dim))
        self.dropout_rate = dropout_rate

        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.JumpingKnowledge.html
        if jumping_knowledge:
            # if setting mode to 'cat', change the channels of the first lins to hidden_conv_dim * heads * n_hidden_conv
            self.jumping_knowledge = JumpingKnowledge(mode='lstm', channels=hidden_conv_dim * heads, num_layers=n_hidden_conv)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        xs = []
        for c in self.convs[:-1]:
            x = c(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.leaky_relu(x, 0.2)
            xs.append(x)

        if self.no_lin:
            return self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
    
        x = self.convs[-1](x, edge_index, edge_attr)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.leaky_relu(x, 0.2)
        xs.append(x)

        if hasattr(self, 'jumping_knowledge'):
            x = self.jumping_knowledge(xs)

        for l in self.lins[:-1]:
            x = l(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.relu(x, 0.2)
        x = self.lins[-1](x)

        return x