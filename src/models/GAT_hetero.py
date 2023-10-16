from torch_geometric.nn import HeteroConv, GATConv, Linear
import torch

class HeteroGAT(torch.nn.Module):

    def __init__(
            self, 
            output_dim_dict, 
            edge_types, 
            n_hidden_conv=2, 
            hidden_conv_dim=32, 
            n_hidden_lin=1, 
            hidden_lin_dim=32, 
            dropout_rate=0.1, 
            *args, 
            **kwargs
        ):

        super().__init__()

        # Apply n conv layers to each edge type
        self.convs = torch.nn.ModuleList()
        self.out_channels_dict = output_dim_dict

        for _ in range(n_hidden_conv):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GATConv((-1, -1), hidden_conv_dim, add_self_loops=False)
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        # Apply n lin layers to each node type
        self.lins = torch.nn.ModuleDict()
        for i in range(n_hidden_lin):
            lin_dict = {}
            for node_type in output_dim_dict.keys():
                if i == 0:
                    lin_dict[node_type] = Linear(hidden_conv_dim, hidden_lin_dim)
                else:
                    lin_dict[node_type] = Linear(hidden_lin_dim, hidden_lin_dim)
            self.lins[str(i)] = torch.nn.ModuleDict(lin_dict)
        
        final_lin_dict = {}
        for node_type, out_channels in output_dim_dict.items():
            if n_hidden_lin == 0:
                final_lin_dict[node_type] = Linear(hidden_conv_dim, out_channels)
            else:
                final_lin_dict[node_type] = Linear(hidden_lin_dim, out_channels)

        self.lins[str(n_hidden_lin)] = torch.nn.ModuleDict(final_lin_dict)

        self.dropout_rate = dropout_rate
       

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: x.relu() for k, x in x_dict.items()}
            x_dict = {k: x.dropout(p=self.dropout_rate, training=self.training) for k, x in x_dict.items()}
        
        for i in range(len(self.lins)-1):
            for node_type in x_dict.keys():
                x_dict[node_type] = self.lins[str(i)][node_type](x_dict[node_type].relu())

        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = self.lins[str(len(self.lins)-1)][node_type](x)

        return out_dict