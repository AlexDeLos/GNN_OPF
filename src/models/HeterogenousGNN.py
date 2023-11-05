from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, Linear, GATv2Conv, GINEConv, RGATConv
import torch
import torch.nn.functional as F
from torch.nn import Sequential, BatchNorm1d, LeakyReLU, ReLU, Dropout
import torch as th
from torch_geometric.nn.models import JumpingKnowledge


class HeteroGNN(torch.nn.Module):
    class_name = "HeteroGNN"
    
    def __init__(
            self,
            output_dim_dict, 
            edge_types,
            n_hidden_conv=1,
            hidden_conv_dim=128,
            n_heads=1,
            n_hidden_lin=3,
            hidden_lin_dim=38,
            dropout_rate=0.4,
            conv_type='GINE', # GAT or GATv2 or SAGE or GINE
            jumping_knowledge='lstm', # max or lstm or mean, None to disable
            hetero_aggr='sum', # sum or mean or max or mul
            *args, 
            **kwargs
        ):

        super().__init__()

        # Apply n conv layers to each edge type
        self.convs = torch.nn.ModuleList()
        self.out_channels_dict = output_dim_dict

        # conv_class = None
        # self.conv_type = conv_type
        # if conv_type == 'GAT':
        #     conv_class = GATConv
        # elif conv_type =='GATv2':
        #     conv_class = GATv2Conv
        # elif conv_type == 'SAGE':
        #     conv_class = SAGEConv
        # elif conv_type == 'GINE':
        #     conv_class = GINEConv
        # else:
        #     raise ValueError(f"conv_type must be 'GAT', 'GATv2', 'GINE' or 'SAGE', not {conv_type}")
        
        # self.n_heads = n_heads
        # for i in range(n_hidden_conv):
        #     conv_dict = {}
        #     for edge_type in edge_types:
        #         if conv_type == 'GAT' or conv_type == 'GATv2':
        #             conv_dict[edge_type] = conv_class((-1, -1), hidden_conv_dim, heads=n_heads, concat=True, edge_dim=-1, dropout=dropout_rate, add_self_loops=False)
        #         elif conv_type == 'SAGE':
        #             conv_dict[edge_type] = conv_class((-1, -1), hidden_conv_dim)
        #         elif conv_type == 'GINE':
        #             # https://github.com/pyg-team/pytorch_geometric/discussions/4607
        #             # need to bring all node types to same dimensionality
        #             if i == 0:
        #                 self.input_lins = torch.nn.ModuleDict()
        #                 for node_type in output_dim_dict.keys():
        #                     self.input_lins[node_type] = Linear(-1, hidden_conv_dim)
        #                 self.input_lins['ext'] = Linear(-1, hidden_conv_dim) # since we miss the key for ext in output_dim_dict
        #             conv_dict[edge_type] = conv_class(
        #             Sequential(
        #                 Linear(hidden_conv_dim, hidden_conv_dim), 
        #                 BatchNorm1d(hidden_conv_dim), 
        #                 # Dropout(dropout_rate),
        #                 ReLU(),
        #                 Linear(hidden_conv_dim, hidden_conv_dim), 
        #                 # Dropout(dropout_rate),
        #                 ReLU(),
        #             ),
        #             edge_dim=-1
        #         )
        #     conv = HeteroConv(conv_dict, aggr=hetero_aggr)
        #     self.convs.append(conv)

        self.convs = torch.nn.ModuleList()
        self.convs.append(RGATConv(2, hidden_conv_dim, num_relations=35, heads=n_heads, num_bases=12, edge_dim=2))
        # self.convs.append(RGATConv(hidden_conv_dim * n_heads, hidden_conv_dim, num_relations=35, heads=n_heads, num_bases=12, edge_dim=2))


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
        self.jk_dict = {}
        if jumping_knowledge == 'max' or jumping_knowledge == 'lstm':
            for node_type in output_dim_dict.keys():
                self.jk_dict[node_type] = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_conv_dim * n_heads, num_layers=n_hidden_conv)
            self.jk_dict['ext'] = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_conv_dim * n_heads, num_layers=n_hidden_conv)
            

    def forward(self, data):
        x, edge_index, edge_attr, edge_type, node_types = data.x, data.edge_index, data.edge_attr, data.edge_type, data.node_type
        # if self.conv_type == 'GINE':
        #     x_dict = {k: self.input_lins[k](x) for k, x in x_dict.items()}
        #     x_dict = {k: F.dropout(x, p=self.dropout_rate, training=self.training) for k, x in x_dict.items()}
                
        # jk_lists_dict = {}
        # for k in x_dict.keys():
        #     jk_lists_dict[k] = []

        # for conv in self.convs:
        #     if self.conv_type == 'SAGE':
        #         x_dict = conv(x_dict, edge_index_dict)
        #     else:
        #         x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        #     x_dict = {k: F.leaky_relu(x, 0.2) for k, x in x_dict.items()}
        #     x_dict = {k: F.dropout(x, p=self.dropout_rate, training=self.training) for k, x in x_dict.items()}
        #     for key in x_dict.keys():
        #         jk_lists_dict[key].append(x_dict[key])
        
        # # if jumping_knowledge is None, we don't use it
        # if self.jumping_knowledge == 'max' or self.jumping_knowledge == 'lstm':
        #     x_dict = {k: self.jk_dict[k](jk_lists_dict[k]) for k in x_dict.keys()}
        # elif self.jumping_knowledge == 'mean':
        #     # since mean is not implemented in pytorch geometric JumpingKnowledge, we do it manually
        #     x_dict = {k: (sum(jk_lists_dict[k]) / len(jk_lists_dict[k])) for k in jk_lists_dict.keys()}

        # node_type: 0 is load, 1 is gen, 2 is load_gen, 3 is ext
        # create a x_dict with all node types, for each node type, create a tensor with all nodes of that type
      
       
        for conv in self.convs:
            x = conv(x, edge_index, edge_type, edge_attr)
            x = F.leaky_relu(x, 0.2)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        
        x_dict = {}
        # create mask for each node type
        x_load = x[node_types == 0]
        x_gen = x[node_types == 1]
        x_load_gen = x[node_types == 2]
        x_ext = x[node_types == 3]
        x_dict['load'] = x_load
        x_dict['gen'] = x_gen
        x_dict['load_gen'] = x_load_gen
        x_dict['ext'] = x_ext
       

        for i in range(len(self.lins)-1):
            for node_type in self.out_channels_dict.keys():
                x_dict[node_type] = self.lins[str(i)][node_type](x_dict[node_type].relu())
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout_rate, training=self.training)


        out_dict = {}
        for node_type in self.out_channels_dict.keys():
            # add relu's for the vm_pu channels (only load second channel/feature [:, 1])
            if node_type == 'load':
                temp_x = self.lins[str(len(self.lins) - 1)][node_type](x_dict[node_type])

                out_dict[node_type] = th.zeros(temp_x.shape, device=temp_x.device)
                out_dict[node_type][:, 0] += temp_x[:, 0]
                out_dict[node_type][:, 1] += th.abs(temp_x[:, 1])
            else:
                out_dict[node_type] = self.lins[str(len(self.lins) - 1)][node_type](x_dict[node_type])

        return out_dict
