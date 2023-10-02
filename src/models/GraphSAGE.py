import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch
from torch.nn import Linear

class GraphSAGE(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            edge_attr_dim=0, 
            hidden_dim=64,
            num_layers=2, 
            dropout=0.2
        ):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        else:
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for i in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.dropout_rate = dropout
        self.linear = Linear(hidden_dim, output_dim)

    # https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html
    # SAGEConv does not take edge attributes
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.linear(x)
        return x

        