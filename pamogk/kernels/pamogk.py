import networkx as nx
import numpy as np
from pamogk.lib.sutils import *
from itertools import product
from tqdm import tqdm


def compute_random_walk_exp(
    G1, pat_idx1, pat_idx2, mutations, beta, bin_based=False, bins=None
):
    assert not (bin_based) or bins is not None
    # arrange all graphs in a list
    gt = nx.Graph()
    # add nodes
    for idx_u, u in enumerate(G1):
        for idx_v, v in enumerate(G1):
            if bin_based:
                label1 = np.where(mutations[pat_idx1, idx_u] >= bins)[0][0]
                label2 = np.where(mutations[pat_idx2, idx_v] >= bins)[0][0]
                if label1 == label2:
                    gt.add_node((u, v), node_label=label1)
            else:
                label1 = mutations[pat_idx1, idx_u] > 0
                label2 = mutations[pat_idx2, idx_v] > 0
                if label1 == label2:
                    gt.add_node((u, v), node_label=label1)

    for (u1, v1), (u2, v2) in product(G1.edges, G1.edges):
        if (u1, u2) in gt and (v1, v2) in gt:
            gt.add_edge((u1, u2), (v1, v2))

    if nx.number_of_nodes(gt) < 2:
        return 0

    A = nx.adjacency_matrix(gt).todense()
    ew, ev = np.linalg.eig(A)

    D = np.zeros((len(ew), len(ew)), dtype=complex)  # @todo: use complex?
    for i in range(len(ew)):
        D[i][i] = np.exp(beta * ew[i])

    exp_D = ev * D * ev.T
    kernel = exp_D.sum()
    if (kernel.real == 0 and np.abs(kernel.imag) < 1e-9) or np.abs(
        kernel.imag / kernel.real
    ) < 1e-9:
        kernel = kernel.real

    return kernel


def kernel_random_walk_exp(
    pat_ids,
    pathway,
    label_key,
    alpha=0.5,
    epsilon=1e-6,
    beta=0.5,
    bin_based=False,
    smooth=False,
    normalization=False,
):
    """
    Parameters
    ----------
    Histogram Kernel
    pat_ids:
        list of patient ids
    pathway:
        pathway networkx graph
    label_key: str
    beta: beta parameter of exponential random walk
    alpha: float
        the smoothing parameter
    epsilon: {1e-6} float
        smoothing converges if the change is lower than epsilon
    normalization: {False} bool
        normalize the kernel matrix such that the diagonal is 1
    """

    num_pat = pat_ids.shape[0]
    pat_ind = {}
    for ind, pid in enumerate(pat_ids):
        pat_ind[pid] = ind
    # extract labels of nodes of graphs
    mutations = np.zeros([num_pat, len(pathway.nodes)], dtype=np.float)
    for idx, nid in enumerate(pathway.nodes):
        nd = pathway.nodes[nid]
        for pid, lb in nd[label_key].items():
            if pid in pat_ind.keys():
                try:
                    _lb = float(lb)
                except ValueError:
                    _lb = 1
                mutations[pat_ind[pid], idx] = _lb

    km = np.zeros((num_pat, num_pat))

    if smooth:
        adj_mat = nx.to_numpy_array(pathway, nodelist=pathway.nodes)
        mutations = smooth(mutations, adj_mat, alpha, epsilon)

    if bin_based:
        bins = arrange_bins(mutations, 20)
        for pat_idx1 in tqdm(pat_ind.values()):
            for pat_idx2 in pat_ind.values():
                km[pat_idx1, pat_idx2] = compute_random_walk_exp(
                    pathway,
                    pat_idx1,
                    pat_idx2,
                    mutations,
                    beta,
                    bin_based=True,
                    bins=bins,
                )
    else:
        for pat_idx1 in tqdm(pat_ind.values()):
            for pat_idx2 in pat_ind.values():
                km[pat_idx1, pat_idx2] = compute_random_walk_exp(
                    pathway, pat_idx1, pat_idx2, mutations, beta
                )

    # normalize the kernel matrix if normalization is true
    if normalization is True:
        km = normalize_kernel_matrix(km)
    return km


def kernel_rbf(
    pat_ids, pathway, label_key, alpha=0.5, epsilon=1e-6, sigma=1, normalization=False
):
    """
    Parameters
    ----------
    Histogram Kernel
    pat_ids:
        list of patient ids
    pathway:
        pathway networkx graph
    label_key: str
    sigma:sigma in the Gaussian RBF kernel
    alpha: float
        the smoothing parameter
    epsilon: {1e-6} float
        smoothing converges if the change is lower than epsilon
    normalization: {False} bool
        normalize the kernel matrix such that the diagonal is 1
    """

    num_pat = pat_ids.shape[0]
    pat_ind = {}
    for ind, pid in enumerate(pat_ids):
        pat_ind[pid] = ind
    # extract labels of nodes of graphs
    mutations = np.zeros([num_pat, len(pathway.nodes)], dtype=np.float)
    for idx, nid in enumerate(pathway.nodes):
        nd = pathway.nodes[nid]
        for pid, lb in nd[label_key].items():
            if pid in pat_ind.keys():
                try:
                    _lb = float(lb)
                except ValueError:
                    _lb = 1
                mutations[pat_ind[pid], idx] = _lb

    # extract the adjacency matrix on the order of nodes we have
    adj_mat = nx.to_numpy_array(pathway, nodelist=pathway.nodes)
    ordered_graph = nx.OrderedGraph()
    ordered_graph.add_nodes_from(pathway.nodes())
    ordered_graph.add_edges_from(sorted(list(pathway.edges())))

    # smooth the mutations through the pathway
    mutations = smooth(mutations, adj_mat, alpha, epsilon)

    bins = arrange_bins(mutations, 20)

    pat_vec = create_hist_matrix(bins, mutations)

    km = RBF(pat_vec)

    # normalize the kernel matrix if normalization is true
    if normalization is True:
        km = normalize_kernel_matrix(km)
    return km


def arrange_bins(
    mutations, 
    num_bins
):

    label_list_sm = []
    for p in range(mutations.shape[0]):

        for idx in range(mutations.shape[1]):

            label_list_sm.append(mutations[p][idx])

    max_lb, min_lb = max(label_list_sm), min(label_list_sm)

    if max_lb - min_lb == 0:
        max_lb = 1

    step_bin = (max_lb - min_lb) / num_bins

    bins = np.arange(min_lb, max_lb + step_bin, step_bin)

    return bins


def create_hist_matrix(
    bins, 
    mutations
):

    num_pat = mutations.shape[0]
    pat_vec = np.zeros([num_pat, len(bins) - 1], dtype=np.float)

    for p in range(mutations.shape[0]):
        vec = []
        vec_hist = 0
        for idx in range(mutations.shape[1]):
            vec.append(mutations[p][idx])
        vec_hist = np.histogram(vec, bins)
        pat_vec[p] = vec_hist[0]
    return pat_vec


def RBF(
    pat_vec, 
    sigma=1
):

    num_pat = pat_vec.shape[0]
    km = np.zeros((num_pat, num_pat))
    for k in range(pat_vec.shape[0]):
        for i in range(pat_vec.shape[0]):

            K = 0

            K = np.dot((pat_vec[k] - pat_vec[i]).T, pat_vec[k] - pat_vec[i])

            K = np.exp(-1.0 * K / (2.0 * sigma * sigma))

            km[k, i] = K
    return km


def kernel(pat_ids, pathway, label_key, alpha=0.5, epsilon=1e-6, normalization=False):
    """
    Parameters
    ----------
    pat_ids:
        list of patient ids
    pathway:
        pathway networkx graph
    alpha: float
        the smoothing parameter
    label_key: str
    epsilon: {1e-6} float
        smoothing converges if the change is lower than epsilon
    normalization: {False} bool
        normalize the kernel matrix such that the diagonal is 1
    """

    num_pat = pat_ids.shape[0]
    pat_ind = {}
    for ind, pid in enumerate(pat_ids):
        pat_ind[pid] = ind
    # extract labels of nodes of graphs
    mutations = np.zeros([num_pat, len(pathway.nodes)], dtype=np.float)
    for idx, nid in enumerate(pathway.nodes):
        nd = pathway.nodes[nid]
        for pid, lb in nd[label_key].items():
            if pid in pat_ind.keys():
                try:
                    _lb = float(lb)
                except ValueError:
                    _lb = 1
                mutations[pat_ind[pid], idx] = _lb

    # extract the adjacency matrix on the order of nodes we have
    adj_mat = nx.to_numpy_array(pathway, nodelist=pathway.nodes)
    ordered_graph = nx.OrderedGraph()
    ordered_graph.add_nodes_from(pathway.nodes())
    ordered_graph.add_edges_from(sorted(list(pathway.edges())))

    # smooth the mutations through the pathway
    mutations = smooth(mutations, adj_mat, alpha, epsilon)
    # get all pairs shortest paths
    all_pairs_sp = nx.all_pairs_shortest_path(ordered_graph)

    km = np.zeros((num_pat, num_pat))

    checked = []
    for src, dsp in all_pairs_sp:  # iterate all pairs shortest paths
        # add source node to checked nodes so we won't check it again in destinations
        checked.append(src)
        # skip if the source is not gene/protein
        if pathway.nodes[src]["type"] != "Protein":
            continue
        # otherwise
        for dst, sp in dsp.items():
            # if destination already checked skip
            if dst in checked:
                continue
            # if the destination is not gene/protein skip
            if pathway.nodes[sp[-1]]["type"] != "Protein":
                continue
            ind = np.isin(pathway.nodes, sp)
            tmp_md = mutations[:, ind]
            # calculate similarities of patients based on the current pathway
            tmp_km = tmp_md @ np.transpose(tmp_md)
            km += tmp_km  # update the main kernel matrix

    # normalize the kernel matrix if normalization is true
    if normalization == True:
        km = normalize_kernel_matrix(km)

    return km


def smooth(md, adj_m, alpha=0.5, epsilon=10 ** -6):
    """
    md: numpy array
        a numpy array of genes of patients indicating which one is mutated or not
    adj_m: numpy array
        the adjacency matrix of the pathway
    alpha: {0.5} float
        the smoothing parameter in range of 0-1
    epsilon: {1e-6} float
        smoothing converges if the change is lower than epsilon
    """
    # since alpha will be together with norm_adj_mat all the time multiply here
    alpha_norm_adj_mat = alpha * adj_m / np.sum(adj_m, axis=0)

    s_md = md
    pre_s_md = md + epsilon + 1

    while np.linalg.norm(s_md - pre_s_md) > epsilon:
        pre_s_md = s_md
        # alpha_norm_adj_mat already includes alpha multiplier
        s_md = (s_md @ alpha_norm_adj_mat) + (1 - alpha) * md

    return s_md


def normalize_kernel_matrix(km):
    kmD = np.array(np.diag(km))
    kmD[kmD == 0] = 1
    D = np.diag(1 / np.sqrt(kmD))
    norm_km = np.linalg.multi_dot([D, km, D.T])  # K_ij / sqrt(K_ii * K_jj)
    return np.nan_to_num(norm_km)  # replace NaN with 0


def main():
    # Create a networkx graph object
    mg = nx.Graph()

    # Add edges to to the graph object
    # Each tuple represents an edge between two nodes
    mg.add_edges_from(
        [
            ("A", "B"),
            ("A", "C"),
            ("C", "D"),
            ("A", "E"),
            ("C", "E"),
            ("D", "B"),
            ("B", "C"),
            ("C", "X"),
        ]
    )

    mg2 = nx.Graph()
    mg2.add_edges_from(
        [
            ("A", "B"),
            ("A", "C"),
            ("C", "D"),
            ("A", "E"),
            ("C", "E"),
            ("D", "B"),
            ("B", "C"),
            ("C", "X"),
        ]
    )

    nx.set_node_attributes(
        mg, {"X": 0, "A": 1, "B": 0, "C": 0, "D": 1, "E": 0}, "label"
    )
    # nx.set_node_attributes(mg, {'X':'Protein', 'A':'Calcium', 'B':'Protein', 'C':'Protein', 'D':'Calcium', 'E':'Protein'}, 'type')
    nx.set_node_attributes(
        mg,
        {
            "X": "Protein",
            "A": "Protein",
            "B": "Protein",
            "C": "Protein",
            "D": "Protein",
            "E": "Protein",
        },
        "type",
    )

    nx.set_node_attributes(
        mg2, {"X": 1, "A": 1, "B": 0, "C": 0, "D": 1, "E": 0}, "label"
    )
    # nx.set_node_attributes(mg2, {'X':'Protein', 'A':'Calcium', 'B':'Protein', 'C':'Protein', 'D':'Calcium', 'E':'Protein'}, 'type')
    nx.set_node_attributes(
        mg2,
        {
            "X": "Protein",
            "A": "Protein",
            "B": "Protein",
            "C": "Protein",
            "D": "Protein",
            "E": "Protein",
        },
        "type",
    )

    # smoothing parameter for the kernel
    alpha = 0.1

    # calculate the kernel using PAMOGK
    # NOTE: this might not be working after some changes to the kernel method
    km = kernel(np.array([0, 1]), mg, alpha=alpha, label_key="label")

    # display the resulting kernel matrix
    print("Kernel matrix calculated by PAMOGK with alpha", alpha)
    print(km)


if __name__ == "__main__":
    main()

