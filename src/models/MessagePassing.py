import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#the-messagepassing-base-class
class MessagePassingGNN(MessagePassing):

    class_name = "MessagePassing"

    def __init__(
            self, 
            input_dim, 
            output_dim, 
            edge_attr_dim, 
            hidden_dim=64,
            num_layers=1,
            dropout_rate=0.1, 
            aggr='mean', 
        ):
        
        super(MessagePassingGNN, self).__init__(aggr=aggr)

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim

        self.node_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()

        if num_layers == 1:
            self.node_layers.append(nn.Linear(input_dim, output_dim))
            self.edge_layers.append(nn.Linear(edge_attr_dim[1], output_dim))

        else:
            # Input layer
            self.node_layers.append(nn.Linear(input_dim, hidden_dim))
            self.edge_layers.append(nn.Linear(edge_attr_dim[1], hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.node_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.edge_layers.append(nn.Linear(hidden_dim, hidden_dim))

            # Output layer
            self.node_layers.append(nn.Linear(hidden_dim, output_dim))
            self.edge_layers.append(nn.Linear(hidden_dim, output_dim))

        # define a mlp for message passing
        self.mlp_message = nn.Sequential(nn.Linear(output_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))
        
        # define a mlp for update
        self.mlp_update = nn.Sequential(nn.Linear(output_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))

        self.dropout_rate = dropout_rate
        
    def forward(self, data):
        # x has shape [num_nodes, input_dim]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_attr_dim]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
        
        # Step 2: Linearly transform node feature matrix.
        for node_layer in self.node_layers[:-1]:
            x = F.relu(node_layer(x))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        for edge_attr_layer in self.edge_layers[:-1]:
            edge_attr = F.relu(edge_attr_layer(edge_attr))
            edge_attr = F.dropout(edge_attr, p=self.dropout_rate, training=self.training)

        x = self.node_layers[-1](x)
        edge_attr = self.edge_layers[-1](edge_attr)
        # Step 4-5: Start propagating messages.
        # propage will call the message function, then the aggregate (i.e. mean) function,
        # and finally the update function.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    
    # build messages to node i from each of its neighbor j
    def message(self, x_i, x_j, edge_attr):
        # x_i contains the node features of the source nodes for each edge. [num_edges, node_hidden_channels]   
        # x_j contains the node features of the target nodes for each edge. [num_edges, node_hidden_channels]
        # edge_attr contains the features of each edge. [num_edges, edge_hidden_channels]
        summed_features =  x_i + x_j + edge_attr
        output = self.mlp_message(summed_features)
        # the message from each edge is indeed the result of an MLP (Multi-Layer Perceptron) applied to the sum of the current node features (x_i), neighboring node features (x_j), and edge features (edge_attr). 
        return output # output is gonna be n of edges x output_dim 

    # aggr_out aggreagtes the messages from the neighbors so we have a message
    # from each edge, and we aggregate messages from neighbors to each node
    # aggr_out has shape [num_nodes, output_dim]
    def update(self, aggr_out):
        # aggr_out contains the output of aggregation. [num_nodes, node_hidden_channels]
        #output = self.mlp_update(aggr_out)
        # Apply the final layer
        return aggr_out
    

    # experiments/ablations:
    # add n of layers in the mlps for message function
    # add biases
    # add skip/residual connections in update function
    # use only edge features and not node features
