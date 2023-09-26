import pandapower.networks as pn
import numpy as np
import torch as th

# net = pn.case5()
# print("\n###### lines ######")
# print(net.line)
# print("\n###### busses ######")
# print(net.bus)
# print("\n###### loads ######")
# print(net.load)
# print("\n###### generators #####")
# print(net.gen)
# print(net.sgen)


# TODO: check pos/neg p_mw in pandapower for correct usage.
#           - Positive for loads means consuming power and negative means generating.
#           - Positive for generators means generating power.

# TODO: enforce min/max constraints on values in the model at the end using sigmoid to bound between e.g. [0,1] and rescale

def physics_loss(network, output, log_loss=False):
    """
    Calculates power imbalances at each node in the graph and sums results.
    Based on loss from https://arxiv.org/abs/2204.07000

    @param network:    Input graph used for the NN model.
                    Expected to contain nodes list and edges between nodes with features:
                        - resistance r over the line
                        - reactance x over the line
    @param output:  Model outputs for each node. Node indices expected to match order in input graph.
                    Expected to contain:
                        - Active power p_mw
                        - Reactive power q_mvar
                        - Volt. mag. vm_pu
                        - Volt. angle va_degree
    @param log_loss: Use normal summed absolute imbalances at each node or a logarithmic version.

    @return:    Returns total power imbalance over all the nodes.
    """
    # Get predicted power levels from the model outputs
    active_imbalance = output[:, 0]
    reactive_imbalance = output[:, 1]

    # Calculate admittance values (conductance, susceptance) from impedance values (edges)
    # edge_att[:, 0] should contain resistances r, edge_att[:, 1] should contain reactances x,
    denom = network.edge_attr[:, 0] * network.edge_attr[:, 0]
    denom += network.edge_attr[:, 1] * network.edge_attr[:, 1]
    conductances = network.edge_attr[:, 0] / denom
    susceptances = -1.0 * network.edge_attr[:, 1] / denom

    # Go over all edges and update the power imbalances for each node accordingly
    # TODO: way to do this with tensors instead of loop?
    for i, x in enumerate(th.transpose(network.edge_index, 0, 1)):
        # x contains node indices [from, to]
        angle_diff = output[x[0], 3] - output[x[1], 3]
        active_imbalance[x[0]] -= np.abs(output[x[0], 2]) * np.abs(output[x[1], 2]) \
                                    * (conductances[i] * np.cos(angle_diff) + susceptances[i] * np.sin(angle_diff))
        reactive_imbalance[x[0]] -= np.abs(output[x[0], 2]) * np.abs(output[x[1], 2]) \
                                    * (conductances[i] * np.sin(angle_diff) - susceptances[i] * np.cos(angle_diff))

    # Use either sum of absolute imbalances or log of squared imbalances
    if log_loss:
        tot_loss = th.sum(np.abs(active_imbalance) + np.abs(reactive_imbalance))
    else:
        tot_loss = th.log(1.0 + th.sum(active_imbalance * active_imbalance + reactive_imbalance * reactive_imbalance))

    return tot_loss
