import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, n_hidden=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        print("here")
        self.convs =  nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(n_hidden):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for c in self.convs[:-1]:
            x = c(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x)

        x = self.convs[-1](x, edge_index)
        return x