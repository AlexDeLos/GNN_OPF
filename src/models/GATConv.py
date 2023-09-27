import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATConv(nn.Module):
    def __init__(self, input_dim, output_dim, edge_attr_dim,
                 n_hidden_gat=2, 
                 hidden_gat_dim=16, 
                 n_hidden_lin=2, 
                 hidden_lin_dim=64, 
                 heads=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()

        self.convs.append(GATConv(input_dim, hidden_gat_dim, heads))
        for _ in range(n_hidden_gat):
            self.convs.append(GATConv(hidden_gat_dim, hidden_gat_dim, heads))

        self.lins.append(nn.Linear(hidden_gat_dim, hidden_lin_dim))
        for _ in range(n_hidden_lin):
            self.lins.append(nn.Linear(hidden_lin_dim, hidden_lin_dim))

        self.lins.append(nn.Linear(hidden_lin_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for c in self.convs:
            x = c(x, edge_index)
            x = F.leaky_relu(x, 0.2)
                
        for l in self.lins[:-1]:
            x = l(x)
            x = F.relu(x, 0.2)
        x = self.lins[-1](x)
        return x