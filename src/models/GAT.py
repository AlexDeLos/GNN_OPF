import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch
from torch.nn import Linear


class GAT(torch.nn.Module):
    # dim_output is the number of classes for classifcation, for regression its the number of nodes
    # hidden_dim and heads are lists of length num_layers
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATv2Conv(in_dim, hidden_dim[-1], heads=heads[0]))
        else:
            index = 0
            self.convs.append(GATv2Conv(in_dim, hidden_dim[0], heads=heads[0]))
            for i in range(num_layers - 2):
                index = i
                self.convs.append(GATv2Conv(hidden_dim[index], hidden_dim[index+1], heads=heads[index]))
            self.convs.append(GATv2Conv(hidden_dim[num_layers - 2], hidden_dim[-1], heads=heads[index+1]))

        self.dropout_rate = 0.6
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.linear = Linear(hidden_dim[-1], out_dim)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.convs[-1](x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.linear(x)
        return x

