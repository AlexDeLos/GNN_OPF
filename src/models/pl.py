import torch as th 
import torch.nn as nn
import torch_geometric.utils as pyg_util
import numpy as np

class ACLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ACLoss, self).__init__()

    def forward(self, output, nodes, edges, attributes):

        active_imbalance = output[:, 0].clone()
        reactive_imbalance = output[:, 1].clone()

        from_nodes = pyg_util.select(output, edges[0], 0)
        to_nodes = pyg_util.select(output, edges[1], 0)

        angle_diffs = th.abs(from_nodes[:, 2] - to_nodes[:, 2])

        act_imb = th.abs(from_nodes[:, 3]) * th.abs(to_nodes[:, 3]) * (attributes[:, 0] * th.cos(angle_diffs) + attributes[:, 1] * th.sin(angle_diffs))  # per edge power flow into/out of from_nodes
        rea_imb = th.abs(from_nodes[:, 3]) * th.abs(to_nodes[:, 3]) * (attributes[:, 0] * th.sin(angle_diffs) - attributes[:, 1] * th.cos(angle_diffs))

        aggr_act_imb = pyg_util.scatter(act_imb, edges[0])  # aggregate all active powers per from_node
        aggr_rea_imb = pyg_util.scatter(rea_imb, edges[0])

        active_imbalance = output[:, 0] - aggr_act_imb  # subtract from power at each node to find imbalance
        reactive_imbalance = output[:, 1] - aggr_rea_imb

        tot_loss = th.sum(th.abs(active_imbalance) + th.abs(reactive_imbalance))
        # print(tot_loss)
        return tot_loss
    
    def reconstruct(inputs, outputs):
        load_indices = (inputs.x[:, 0] == 1).nonzero()
        gen_indices = (inputs.x[:, 1] == 1).nonzero()
        ext_indices = (inputs.x[:, 2] == 1).nonzero()
        none_indices = (inputs.x[:, 3] == 1).nonzero()

        load_values = pyg_util.select(inputs, load_indices, 0)
        gen_values = pyg_util.select(inputs, gen_indices, 0)
        ext_values = pyg_util.select(inputs, ext_indices, 0)
        none_values = pyg_util.select(inputs, none_indices, 0)

        





        
