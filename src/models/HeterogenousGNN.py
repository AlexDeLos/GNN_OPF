from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, Linear, GATv2Conv, GINEConv
import torch
import torch.nn.functional as F
from torch.nn import Sequential, BatchNorm1d, LeakyReLU, ReLU

class HeteroGNN(torch.nn.Module):
    class_name = "HeteroGNN"
    
    def __init__(
            self,
            output_dim_dict, 
            edge_types, 
            n_hidden_conv=3, 
            hidden_conv_dim=64,
            n_heads=1, 
            n_hidden_lin=2, 
            hidden_lin_dim=64, 
            dropout_rate=0.3,
            conv_type='GINE', # GAT or GATv2 or SAGE or GINE
            jumping_knowledge=True,
            hetero_aggr='mean',
            *args, 
            **kwargs
        ):

        super().__init__()

        # Apply n conv layers to each edge type
        self.convs = torch.nn.ModuleList()
        self.out_channels_dict = output_dim_dict

        conv_class = None
        self.conv_type = conv_type
        if conv_type == 'GAT':
            conv_class = GATConv
        elif conv_type =='GATv2':
            conv_class = GATv2Conv
        elif conv_type == 'SAGE':
            conv_class = SAGEConv
        elif conv_type == 'GINE':
            conv_class = GINEConv
        else:
            raise ValueError(f"conv_type must be 'GAT', 'GATv2' or 'SAGE', not {conv_type}")
        
        self.n_heads = n_heads
        for i in range(n_hidden_conv):
            conv_dict = {}
            for edge_type in edge_types:
                if conv_type == 'GAT' or conv_type == 'GATv2':
                    conv_dict[edge_type] = conv_class((-1, -1), hidden_conv_dim, heads=n_heads, concat=True, edge_dim=-1, dropout=0.2, add_self_loops=False)
                elif conv_type == 'SAGE':
                    conv_dict[edge_type] = conv_class((-1, -1), hidden_conv_dim)
                elif conv_type == 'GINE':
                    # https://github.com/pyg-team/pytorch_geometric/discussions/4607
                    # need to bring all node type to same dimensionality
                    if i == 0:
                        self.input_lins = torch.nn.ModuleDict()
                        for node_type in output_dim_dict.keys():
                            self.input_lins[node_type] = Linear(-1, hidden_conv_dim)
                        self.input_lins['ext'] = Linear(-1, hidden_conv_dim) # since we miss the key for ext in output_dim_dict
                    conv_dict[edge_type] = conv_class(
                    Sequential(
                        Linear(hidden_conv_dim, hidden_conv_dim), 
                        BatchNorm1d(hidden_conv_dim), 
                        ReLU(),
                        Linear(hidden_conv_dim, hidden_conv_dim), 
                        ReLU(),
                    ),
                    edge_dim=-1
                )
            conv = HeteroConv(conv_dict, aggr=hetero_aggr)
            self.convs.append(conv)

        # Apply n lin layers to each node type
        self.lins = torch.nn.ModuleDict()
        for i in range(n_hidden_lin):
            lin_dict = {}
            for node_type in output_dim_dict.keys():
                if i == 0:
                    lin_dict[node_type] = Linear(hidden_conv_dim * n_heads, hidden_lin_dim)
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
        self.jumping_knowledge = jumping_knowledge
       

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        if self.conv_type == 'GINE':
            x_dict = {k: self.input_lins[k](x) for k, x in x_dict.items()}
            x_dict = {k: F.dropout(x, p=self.dropout_rate, training=self.training) for k, x in x_dict.items()}
                
        jumping_knowledge_dict = {}
        for k in x_dict.keys():
            jumping_knowledge_dict[k] = []

        for conv in self.convs:
            if self.conv_type == 'SAGE':
                x_dict = conv(x_dict, edge_index_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

            x_dict = {k: F.leaky_relu(x, 0.2) for k, x in x_dict.items()}
            x_dict = {k: F.dropout(x, p=self.dropout_rate, training=self.training) for k, x in x_dict.items()}
            for key in x_dict.keys():
                jumping_knowledge_dict[key].append(x_dict[key])
        
        if self.jumping_knowledge:
            # mean of the hidden states of each conv layer
            x_dict = {k: (sum(jumping_knowledge_dict[k]) / len(jumping_knowledge_dict[k])) for k in jumping_knowledge_dict.keys()}
        
        for i in range(len(self.lins)-1):
            for node_type in self.out_channels_dict.keys():
                x_dict[node_type] = self.lins[str(i)][node_type](x_dict[node_type].relu())

        out_dict = {}
        for node_type in self.out_channels_dict.keys():
            out_dict[node_type] = self.lins[str(len(self.lins)-1)][node_type](x_dict[node_type])

        return out_dict