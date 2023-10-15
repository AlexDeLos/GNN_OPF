import torch as th
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class HeteroGNN(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 edge_attr,
                 n_hidden_conv=0, 
                 hidden_conv_dim=16, 
                 n_hidden_lin=2, 
                 hidden_lin_dim=16,  
                 dropout=0.1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()

        for in_channel, out_channel in zip(in_channels, out_channels):
            self.convs.append(SAGEConv(in_channel, out_channel))
            self.lins.append(nn.Linear(in_channel, out_channel))

    def forward(self, data):
        xs = []
        ys = []
        for i, key in enumerate(data.node_types):
            x = data[key].x
            edge_index = data['connects'].edge_index
            x = self.convs[i](x, edge_index)
            y = self.lins[i](x)
            xs.append(x)
            ys.append(y)
        return xs, ys