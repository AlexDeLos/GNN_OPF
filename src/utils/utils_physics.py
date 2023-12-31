import math
import pandapower.plotting as ppl
import torch as th
import torch_geometric.utils as pyg_util
from torch_geometric.utils.convert import from_networkx

# Create the data needed for the physics loss
# return a torch_geometric.data.Data object for each instance
def create_physics_data_instance(graph, y_bus, missing, volt):
    """
    Converts a PandaPower graph to a NetworkX graph and creates a node feature matrix for the graph.

    Args:
        graph (pandapowerNet): The PandaPower graph to convert.
        y_bus (pandas.DataFrame): The Y-bus matrix for the graph.
        missing (bool): Whether to include missing data in the node feature matrix.
        volt (bool): Whether to include voltage data in the node feature matrix.

    Returns:
        networkx.Graph: The converted NetworkX graph.
    """
    # Convert PandaPower graph to NetworkX graph and set it to be directed (for directed transformer edges further down)
    g = ppl.create_nxgraph(graph, include_trafos=True)
    g = g.to_directed()

    # https://pandapower.readthedocs.io/en/latest/elements/gen.html
    gen = graph.gen[['bus', 'p_mw', 'vm_pu']]
    gen.rename(columns={'p_mw': 'p_mw_gen'}, inplace=True)
    gen['is_gen'] = 1
    gen.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/sgen.html
    # Note: multiple static generators can be attached to 1 bus!
    sgen = graph.sgen[['bus', 'p_mw', 'q_mvar']]
    sgen.rename(columns={'p_mw': 'p_mw_sgen'}, inplace=True)
    sgen.rename(columns={'q_mvar': 'q_mvar_sgen'}, inplace=True)
    sgen = sgen.groupby('bus')[['p_mw_sgen', 'q_mvar_sgen']].sum()  # Already resets index
    sgen['is_sgen'] = 1

    # https://pandapower.readthedocs.io/en/latest/elements/load.html
    load = graph.load[['bus', 'p_mw', 'q_mvar']]
    load.rename(columns={'p_mw': 'p_mw_load'}, inplace=True)
    load.rename(columns={'q_mvar': 'q_mvar_load'}, inplace=True)
    load['is_load'] = 1
    load.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/ext_grid.html
    ext = graph.ext_grid[['bus', 'vm_pu', 'va_degree']]
    ext.rename(columns={'vm_pu': 'vm_pu_ext'}, inplace=True)
    ext_degree = ext.loc[0, 'va_degree']
    # if ext_degree != 30.0:
    #     print('ext_degree')
    #     print(ext_degree)
    ext['is_ext'] = 1
    ext.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/shunt.html
    shunt = graph.shunt[['bus', 'q_mvar', 'step']]
    shunt['b_pu_shunt'] = shunt['q_mvar'] * shunt['step'] / graph.sn_mva
    shunt.rename(columns={'q_mvar': 'q_mvar_shunt'}, inplace=True)
    del shunt['step']
    shunt.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/bus.html
    node_feat = graph.bus[['vn_kv']]

    # make sure all nodes (bus, gen, load) have the same number of features (namely the union of all features)
    node_feat = node_feat.merge(gen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(sgen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(load, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(ext, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(shunt, left_index=True, right_index=True, how='outer')

    # fill missing feature values with 0
    node_feat.fillna(0.0, inplace=True)
    node_feat['vm_pu'] = node_feat['vm_pu'] + node_feat['vm_pu_ext']
    node_feat['p_mw'] = node_feat['p_mw_load'] - node_feat['p_mw_gen'] - node_feat['p_mw_sgen']
    node_feat['q_mvar'] = node_feat['q_mvar_load'] + node_feat['q_mvar_shunt'] - node_feat['q_mvar_sgen']

    # static generators are modeled as loads in PandaPower
    node_feat['is_load'] = (node_feat['is_sgen'] != 0) | (node_feat['is_load'] != 0)

    del node_feat['vm_pu_ext']
    del node_feat['p_mw_gen']
    del node_feat['p_mw_sgen']
    del node_feat['p_mw_load']
    del node_feat['q_mvar_load']
    del node_feat['q_mvar_sgen']
    del node_feat['q_mvar_shunt']
    del node_feat['is_sgen']

    # remove duplicate columns/indices
    node_feat = node_feat[~node_feat.index.duplicated(keep='first')]
    node_feat['is_none'] = (node_feat['is_gen'] == 0) & (node_feat['is_load'] == 0) & (node_feat['is_ext'] == 0)
    node_feat['is_none'] = node_feat['is_none'].astype(float)
    node_feat = node_feat[['is_load', 'is_gen', 'is_ext', 'is_none', 'p_mw', 'q_mvar', 'va_degree', 'vm_pu', 'b_pu_shunt']]
    zero_check = node_feat[(node_feat['is_load'] == 0) & (node_feat['is_gen'] == 0) & (node_feat['is_ext'] == 0) & (
                node_feat['is_none'] == 0)]

    if not zero_check.empty:
        print("zero check failed")
        print(node_feat)
        print("zero check results")
        print(zero_check)
        quit()

    for node in node_feat.itertuples():
        # set each node features
        g.nodes[node.Index]['x'] = [float(node.is_load),
                                    float(node.is_gen),
                                    float(node.is_ext),
                                    float(node.is_none),
                                    float(node.p_mw / graph.sn_mva),
                                    float(node.q_mvar / graph.sn_mva),
                                    float(node.vm_pu),
                                    float(node.va_degree)]

        if missing:
            if node.is_load or node.is_none:
                g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
            if node.is_gen and not node.is_load:
                g.nodes[node.Index]['y'] = [float(y_bus['q_mvar'][node.Index] / graph.sn_mva),
                            float(y_bus['va_degree'][node.Index])]
            if node.is_ext:
                g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index] / graph.sn_mva),
                            float(y_bus['q_mvar'][node.Index]) / graph.sn_mva]
        elif volt:
            g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
             
        else:
            g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index] / graph.sn_mva),
                                    float(y_bus['q_mvar'][node.Index] / graph.sn_mva),
                                    float(y_bus['vm_pu'][node.Index]),
                                    float(y_bus['va_degree'][node.Index])]

    for edges in graph.line.itertuples():
        # Calculate line admittance from impedance and convert to per-unit system
        r_tot = float(edges.r_ohm_per_km) * float(edges.length_km)
        x_tot = float(edges.x_ohm_per_km) * float(edges.length_km)
        conductance_line, susceptance_line = impedance_to_admittance(r_tot, x_tot, graph.bus['vn_kv'][edges.from_bus], graph.sn_mva)
        g.edges[edges.from_bus, edges.to_bus, ('line', edges.Index)]['edge_attr'] = [float(conductance_line), float(susceptance_line)]
        g.edges[edges.to_bus, edges.from_bus, ('line', edges.Index)]['edge_attr'] = [float(conductance_line), float(susceptance_line)]

    for trafos in graph.trafo.itertuples():
        # Calculate trafo impedance using low voltage side as base voltage
        # Assumes simplified trafo model (no vkr, iron loss, tap)
        r_tot = 0.0
        x_tot = (trafos.vk_percent / 100.0) * (trafos.vn_lv_kv ** 2) / trafos.sn_mva
        conductance, susceptance = impedance_to_admittance(r_tot, x_tot, trafos.vn_lv_kv, graph.sn_mva)
        g.edges[trafos.hv_bus, trafos.lv_bus, ('trafo', trafos.Index)]['edge_attr'] = [float(conductance), float(susceptance)]
        g.edges[trafos.lv_bus, trafos.hv_bus, ('trafo', trafos.Index)]['edge_attr'] = [float(conductance), float(susceptance)]

    return from_networkx(g)


def impedance_to_admittance(r_ohm, x_ohm, base_volt, rated_power, per_unit_conversion=True):
    if per_unit_conversion:
        z_base = math.pow(base_volt, 2) / rated_power  # Z_base used to convert impedance to per-unit
        r_tot = r_ohm / z_base  # Convert to per unit metrics before converting to admittance
        x_tot = x_ohm / z_base
    else:
        r_tot = r_ohm
        x_tot = x_ohm
    denom = math.pow(r_tot, 2) + math.pow(x_tot, 2)
    conductance = r_tot / denom
    susceptance = -x_tot / denom
    return conductance, susceptance


def physics_loss(network, output, log_loss=True, device='cpu'):
    """
    Calculates power imbalances at each node in the graph and sums results.
    Based on loss from https://arxiv.org/abs/2204.07000

    @param network:    Input graph used for the NN model.
                    Expected to contain nodes list and edges between nodes with features:
                        - conductance over the line
                        - susceptance over the line
                        - susceptance of line shunt
    @param output:  Model outputs for each node. Node indices expected to match order in input graph.
                    Expected to contain:
                        - Active power p_mw
                        - Reactive power q_mvar
                        - Volt. mag. vm_pu
                        - Volt. angle va_degree
    @param log_loss: Use normal summed absolute imbalances at each node or a logarithmic version.

    @return:    Returns total power imbalance over all the nodes.
    """
    # Combine the fixed input values and predicted missing values
    combined_output = th.zeros(output.shape).to(device) 
    inp_idx = {'is_load': 0, 'is_gen': 1, 'is_ext': 2, 'is_none': 3, 'p_mw': 4, 'q_mvar': 5, 'vm_pu': 6, 'va_degree': 7}
    out_idx = {'p_mw': 0, 'q_mvar': 1, 'vm_pu': 2, 'va_degree': 3}

    # slack bus:
    idx_list = (network.x[:, inp_idx['is_ext']] > 0.5)  # get slack node id's
    combined_output[idx_list, out_idx['vm_pu']] += network.x[idx_list, inp_idx['vm_pu']]  # Add fixed vm_pu from input
    combined_output[idx_list, out_idx['va_degree']] += network.x[idx_list, inp_idx['va_degree']]  # Add fixed va_degree from input
    combined_output[idx_list, out_idx['p_mw']] += output[idx_list, out_idx['p_mw']]  # Add predicted p_mw
    combined_output[idx_list, out_idx['q_mvar']] += output[idx_list, out_idx['q_mvar']]  # Add predicted q_mvar

    # generator + load busses:
    idx_list = (th.logical_and(network.x[:, inp_idx['is_load']] > 0.5, network.x[:, inp_idx['is_gen']] > 0.5))  # get generator + load node id's

    combined_output[idx_list, out_idx['p_mw']] += network.x[idx_list, inp_idx['p_mw']]  # Add fixed p_mw from input (already contains value of load p_mw - gen p_mw, so we add instead of subtract)
    combined_output[idx_list, out_idx['vm_pu']] += network.x[idx_list, inp_idx['vm_pu']]  # Add fixed vm_pu from input (should be same for both load and gen)
    combined_output[idx_list, out_idx['q_mvar']] += output[idx_list, out_idx['q_mvar']]  # Add predicted q_mvar
    combined_output[idx_list, out_idx['va_degree']] += output[idx_list, out_idx['va_degree']]  # Add predicted va_degree

    # generator:
    idx_list = (th.logical_and(network.x[:, inp_idx['is_load']] < 0.5, network.x[:, inp_idx['is_gen']] > 0.5))  # get generator (not gen + load) node id's
    combined_output[idx_list, out_idx['p_mw']] += network.x[idx_list, inp_idx['p_mw']]  # Add fixed p_mw from input (already set to neg. in data gen.)
    combined_output[idx_list, out_idx['vm_pu']] += network.x[idx_list, inp_idx['vm_pu']]  # Add fixed vm_pu from input
    combined_output[idx_list, out_idx['q_mvar']] += output[idx_list, out_idx['q_mvar']]  # Add predicted q_mvar
    combined_output[idx_list, out_idx['va_degree']] += output[idx_list, out_idx['va_degree']]  # Add predicted va_degree

    # load + none types (modeled as 0 power demand loads):
    load_no_gen = th.logical_and(network.x[:, inp_idx['is_load']] > 0.5, network.x[:, inp_idx['is_gen']] < 0.5)
    idx_list = (th.logical_or(th.logical_and(load_no_gen, network.x[:, inp_idx['is_ext']] < 0.5), network.x[:, inp_idx['is_none']] > 0.5))  # get load + none node id's
    combined_output[idx_list, out_idx['p_mw']] += network.x[idx_list, inp_idx['p_mw']]  # Add fixed p_mw from input
    combined_output[idx_list, out_idx['q_mvar']] += network.x[idx_list, inp_idx['q_mvar']]  # Add fixed q_mvar from input
    combined_output[idx_list, out_idx['vm_pu']] += output[idx_list, out_idx['vm_pu']]  # Add predicted vm_pu
    combined_output[idx_list, out_idx['va_degree']] += output[idx_list, out_idx['va_degree']] # Add predicted va_degree

    # calculate active and reactive power from the voltage magnitude and angles
    power_values = power_from_voltages(network, combined_output[:, [out_idx['vm_pu'], out_idx['va_degree']]], angles_are_radians=False)

    # subtract from power at each node to find imbalance. negate power output values due to pos/neg conventions for loads/gens
    active_imbalance = power_values[:, 0] - combined_output[:, out_idx['p_mw']]
    reactive_imbalance = power_values[:, 1] - combined_output[:, out_idx['q_mvar']]

    # Use either sum of absolute imbalances or log of squared imbalances
    if log_loss:
        tot_loss = th.log(1.0 + th.sum(active_imbalance * active_imbalance + reactive_imbalance * reactive_imbalance))
    else:
        tot_loss = th.sum(th.abs(active_imbalance) + th.abs(reactive_imbalance))

    return tot_loss


def power_from_voltages(network, voltages, angles_are_radians=False):
    """
    Calculates the active and reactive power for each node depending on the given network and vm_pu and va_degree values.

    Args:
        network: power grid network with all nodes, edges
        voltages: 2d tensor of [vm_pu, va_degree] for the nodes

    Returns:
        2d tensor of [p_mw, q_mvar] for each node
    """
    # negative conductances and susceptances for the off-diagonal elements of the nodal admittance matrix
    conductances = -1.0 * network.edge_attr[:, 0]
    susceptances = -1.0 * network.edge_attr[:, 1]

    out_idx = {'vm_pu': 0, 'va_degree': 1}
    # combine node features with corresponding edges
    from_nodes = pyg_util.select(voltages, network.edge_index[0], 0)  # list of duplicated node outputs based on edges
    to_nodes = pyg_util.select(voltages, network.edge_index[1], 0)
    if angles_are_radians:
        angle_diffs = (from_nodes[:, out_idx['va_degree']] - to_nodes[:, out_idx['va_degree']])  # list of angle differences for all edges
    else:
        angle_diffs = (from_nodes[:, out_idx['va_degree']] - to_nodes[:, out_idx['va_degree']]) * math.pi / 180.0  # list of angle differences for all edges

    # calculate incoming/outgoing values based on the edges connected to each node and the node's + neighbour's values
    act_pow_lines = th.abs(from_nodes[:, out_idx['vm_pu']]) * th.abs(to_nodes[:, out_idx['vm_pu']]) * (conductances * th.cos(angle_diffs) + susceptances * th.sin(angle_diffs))  # per edge power flow into/out of from_nodes
    rea_pow_lines = th.abs(from_nodes[:, out_idx['vm_pu']]) * th.abs(to_nodes[:, out_idx['vm_pu']]) * (conductances * th.sin(angle_diffs) - susceptances * th.cos(angle_diffs))

    act_pow_node = pyg_util.scatter(act_pow_lines, network.edge_index[0])  # aggregate all active powers for each node
    rea_pow_node = pyg_util.scatter(rea_pow_lines, network.edge_index[0])  # same for reactive

    # add diagonal (self-admittance) elements of each node as well (angle diff is 0; only cos sections have an effect)
    act_pow_node += voltages[:, out_idx['vm_pu']] * voltages[:, out_idx['vm_pu']] * pyg_util.scatter(network.edge_attr[:, 0], network.edge_index[0])
    # for reactive self-admittance we also take into account the shunt reactances and not only line reactances
    rea_pow_node += voltages[:, out_idx['vm_pu']] * voltages[:, out_idx['vm_pu']] * (-1.0 * (pyg_util.scatter(network.edge_attr[:, 1], network.edge_index[0])))

    # Return -1 times the values, since we use negative values for gens and positive for loads
    return th.cat(((-1.0 * act_pow_node).reshape(-1, 1), (-1.0 * rea_pow_node).reshape(-1, 1)), 1)
