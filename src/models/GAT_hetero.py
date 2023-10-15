from torch_geometric.nn import HeteroConv, GATConv, Linear
import torch

class HeteroGAT(torch.nn.Module):

    def __init__(self, hidden_channels, num_layers, edge_types, out_channels_dict):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.out_channels_dict = out_channels_dict
        print(edge_types)
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GATConv((-1, -1), hidden_channels, add_self_loops=False)
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        self.lins = torch.nn.ModuleDict()
        for node_type, out_channels in out_channels_dict.items():
            self.lins[node_type] = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: x.relu() for k, x in x_dict.items()}
            x_dict = {k: x.dropout(p=0.1, training=self.training) for k, x in x_dict.items()}
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = self.lins[node_type](x)
        return out_dict