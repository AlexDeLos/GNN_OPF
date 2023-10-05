import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch
import torch.nn as nn

class GraphSAGE(torch.nn.Module):

    class_name = "GraphSAGE"

    def __init__(
            self, 
            input_dim, 
            output_dim, 
            edge_attr_dim=0, # not used by SAGEConv, necessary for compatibility with models that use edge attributes
            n_hidden_conv=2, 
            hidden_conv_dim=64, 
            n_hidden_lin=1, 
            hidden_lin_dim=64,  
            dropout=0.1,
        ):
        
        super().__init__()
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels=input_dim, out_channels=hidden_conv_dim))
        for _ in range(n_hidden_conv):
            self.convs.append(SAGEConv(in_channels=hidden_conv_dim, out_channels=hidden_conv_dim))
        
        self.lins.append(nn.Linear(hidden_conv_dim, hidden_lin_dim))
        for _ in range(n_hidden_lin):
            self.lins.append(nn.Linear(hidden_lin_dim, hidden_lin_dim))

        self.lins.append(nn.Linear(hidden_lin_dim, output_dim))

        self.dropout_rate = dropout

    # https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html
    # SAGEConv does not take edge attributes
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for c in self.convs:
            x = c(x=x, edge_index=edge_index)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.elu(x)

        for l in self.lins[:-1]:
            x = l(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.elu(x)

        x = self.lins[-1](x)

        return x

        