import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import JumpingKnowledge

class MessagePassingGNN(nn.Module):
    class_name = "MessagePassing"

    def __init__(
            self, 
            input_dim, 
            output_dim, 
            edge_attr_dim,
            n_message_passing_layers=2,
            hidden_msp_dim=256,
            n_lin_layers=1,
            hidden_lin_dim=128,
            dropout_rate=0.1, 
            aggr='mean', 
            jumping_knowledge=True,
            no_lin=False
        ):
        
        super().__init__()

        self.n_lin_layers = n_lin_layers
        self.input_dim = input_dim
        self.edge_attr_dim = edge_attr_dim
        self.hidden_lin_dim = hidden_lin_dim

        self.node_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()

        self.node_layers.append(nn.Linear(input_dim, hidden_lin_dim))
        self.edge_layers.append(nn.Linear(edge_attr_dim, hidden_lin_dim))

        self.dropout_rate = dropout_rate

        for _ in range(n_lin_layers - 1):
            self.node_layers.append(nn.Linear(hidden_lin_dim, hidden_lin_dim))
            self.edge_layers.append(nn.Linear(hidden_lin_dim, hidden_lin_dim))

        self.n_message_passing_layers = n_message_passing_layers

        self.message_passing_layers = nn.ModuleList()

        input_msp = hidden_lin_dim
        output_msp = hidden_lin_dim

        for i in range(n_message_passing_layers):
            if i == n_message_passing_layers - 1 and no_lin:
                output_msp = output_dim
        
            self.message_passing_layers.append(
                MessagePassingLayer(
                    input_dim=input_msp,
                    hidden_dim=hidden_msp_dim,
                    output_dim=output_msp,
                    aggr=aggr, 
                )
            )
        self.no_lin = no_lin
        self.final_lin = nn.Linear(hidden_lin_dim, output_dim)

        if jumping_knowledge:
            self.jumping_knowledge = JumpingKnowledge(mode='max', channels=hidden_lin_dim, num_layers=n_message_passing_layers)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        for node_layer in self.node_layers[:-1]:
            x = F.relu(node_layer(x))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        for edge_attr_layer in self.edge_layers[:-1]:
            edge_attr = F.relu(edge_attr_layer(edge_attr))
            edge_attr = F.dropout(edge_attr, p=self.dropout_rate, training=self.training)
        
        x = self.node_layers[-1](x)
        
        edge_attr = self.edge_layers[-1](edge_attr)

        xs = []
        xs.append(x)

        for layer in self.message_passing_layers:
            x = layer(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            xs.append(x)

        if hasattr(self, 'jumping_knowledge'):
            x = self.jumping_knowledge(xs)

        if not(self.no_lin):
            x = self.final_lin(x)
    
        return x

    
# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#the-messagepassing-base-class
class MessagePassingLayer(MessagePassing):
    def __init__(
            self, 
            input_dim,
            hidden_dim,
            output_dim,
            aggr='mean', 
        ):
        
        super().__init__(aggr=aggr)

        # define a mlp to build messages
        self.mlp_message = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
        # define a mlp for update node representations
        self.mlp_update = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index, edge_attr):
        # propage will call the message function, then the aggregate (i.e. mean) function, and finally the update function.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    
    # build messages to node i from each of its neighbor j
    def message(self, x_i, x_j, edge_attr):
        # x_i contains the node features of the source nodes for each edge. [num_edges, hidden_lin_dim]   
        # x_j contains the node features of the target nodes for each edge. [num_edges, hidden_lin_dim]
        # edge_attr contains the features of each edge. [num_edges, edge_hidden_channels] 

        # here's there's a lot of freedom, we can sum, concatenate, take mean, mlps, etc.
        x_j = self.mlp_message(x_j)
        edge_attr = self.mlp_message(edge_attr)
        return x_j + edge_attr # [num_edges, node_hidden_channels]

    # aggr_out aggregates the messages from the neighbors so we have a message
    # from each edge, and we aggregate messages from neighbors to each node
    # aggr_out has shape [num_nodes, node_hidden_channels]
    def update(self, aggr_out, x):
        # aggr_out contains the output of aggregation. [num_nodes, node_hidden_channels]
        # x contains the old node representations we want to update, [num_nodes, node_hidden_channels]
        # we add the old node representations to the aggregated messages and pass them through a mlp 
        x = self.mlp_update(x)
        aggr_out = self.mlp_update(aggr_out)
        return x + aggr_out # [num_nodes, node_hidden_channels]