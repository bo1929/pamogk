import numpy as np


def mark_label_on_pathways(name, pid, pw_map, gene_id_list, label=1):
    """Marks given genes to the pathways

    Parameters
    ----------
    name: str
    pid: int
        patient id
    pw_map: map of networkx graphs of pathways
        patient label mapping
    gene_id_list: list of list of string
        uniprot gene id list of genes
    label: int
        the label which will be assigned to found genes in pathways - default value is 1
    """
    label_field = f"label-{name}"
    gene_ids = [uid for a in gene_id_list for uid in a]
    for pw in pw_map.values():  # for each pathway
        for n in pw.nodes():
            nd = pw.nodes[n]
            if label_field not in nd:
                pw.add_node(n, **{label_field: {}})
            if np.any([g in nd["entrezids"] for g in gene_ids]):
                nd[label_field][pid] = label


def mark_label_on_communities(name, pid, comm_map, gene_id_list, label=1):
    """Marks given genes to the communities

    Parameters
    ----------
    name: str
    pid: int
        patient id
    comm_map: map of networkx graphs of communities
        patient label mapping
    gene_id_list: list of list of string
        gene entrez gene id list of genes
    label: int
        the label which will be assigned to found genes in pathways - default value is 1
    """
    label_field = f"label-{name}"
    gene_ids = [eid for a in gene_id_list for eid in a]
    for comm in comm_map.values():
        for n in comm.nodes():
            nd = comm.nodes[n]
            if label_field not in nd:
                comm.add_node(n, **{label_field: {}})
            if np.any([int(g) in nd["entrezids"] for g in gene_ids]):
                nd[label_field][pid] = label


def mark_cont_label_on_pathways(name, pid, comm_map, uni_ids, gene_vals):
    """Marks given genes and their normalized expressions to the pathways

    Parameters
    ----------
    name: str
    pid: int
            patient id
    comm_map: map of networkx graphs of pathways
            patient label mapping
    uni_ids: list of list of string
            uniprot gene id list of genes
    gene_vals: :obj:`numpy.ndarray`
            the values of genes which will be assigned to found genes in pathways
    """
    label_field = f"label-{name}"
    for pw in comm_map.values():
        for n in pw.nodes():
            nd = pw.nodes[n]
            if label_field not in nd:
                pw.add_node(n, **{label_field: {}})
            intersect_values = gene_vals[
                [len(set(nd["uniprotids"]).intersection(g)) > 0 for g in uni_ids]
            ]
            if len(intersect_values) > 0:
                print(len(intersect_values))
                if "oe" in name:
                    nd[label_field][pid] = max(0, max(intersect_values))
                elif "ue" in name:
                    nd[label_field][pid] = min(0, min(intersect_values))
                elif "abs" in name:
                    nd[label_field][pid] = max(
                        intersect_values.max(), intersect_values.min(), key=abs
                    )


def mark_cont_label_on_communities(name, pid, comm_map, entrez_ids, gene_vals):
    """Marks given genes and their normalized expressions to the communities

    Parameters
    ----------
    name: str
    pid: int
            patient id
    comm_map: map of networkx graphs of pathways
            patient label mapping
    entrez_ids: list of list of string
            entrez gene id list of genes
    gene_vals: :obj:`numpy.ndarray`
            the values of genes which will be assigned to found genes in pathways
    """
    label_field = f"label-{name}"
    for comm in comm_map.values():
        for n in comm.nodes():
            nd = comm.nodes[n]
            if label_field not in nd:
                comm.add_node(n, **{label_field: {}})
            intersect_values = gene_vals[
                [
                    len(set(nd["entrezids"]).intersection([int(g)])) > 0
                    for g in entrez_ids
                ]
            ]
            if len(intersect_values) > 0:
                if "oe" in name:
                    nd[label_field][pid] = max(0, max(intersect_values))
                elif "ue" in name:
                    nd[label_field][pid] = min(0, min(intersect_values))
                elif "abs" in name:
                    nd[label_field][pid] = max(
                        intersect_values.max(), intersect_values.min(), key=abs
                    )


def mark_extra_label_on_pathways(name, pid, pw_map, old_label_name, threshold=1.96):
    """Marks new labels on pathways using old_labels

    Parameters
    ----------
    name: string
        new label name
    pid: int
        patient id
    pw_map: map of networkx graphs of pathways
        patient label mapping
    old_label_name: string
        old label name that will be used for new label
    threshold: float
        threshold of new label. If abs(old_label)<threshold then new label=0 same otherwise - default value is 1.96
    """
    label_field = f"label-{name}"
    old_label_field = f"label-{old_label_name}"
    oe_label_field = "label-oe"
    ue_label_field = "label-ue"
    for pw in pw_map.values():  # for each pathway
        for n in pw.nodes():
            nd = pw.nodes[n]
            if label_field not in nd:
                pw.add_node(n, **{label_field: {}})
            if name == "onekernel":
                if pid in nd[oe_label_field].keys():
                    nd[label_field][pid] = nd[oe_label_field][pid]
                elif pid in nd[ue_label_field].keys():
                    nd[label_field][pid] = nd[ue_label_field][pid]
            else:
                if (
                    pid in nd[old_label_field].keys()
                    and abs(nd[old_label_field][pid]) < threshold
                ):
                    nd[label_field][pid] = 0
                elif pid in nd[old_label_field].keys():
                    nd[label_field][pid] = nd[old_label_field][pid]


def mark_extra_label_on_communities(
    name, pid, comm_map, old_label_name, threshold=1.96
):
    """Marks new labels on pathways using old_labels

    Parameters
    ----------
    name: string
        new label name
    pid: int
        patient id
    comm_map: map of networkx graphs of pathways
        patient label mapping
    old_label_name: string
        old label name that will be used for new label
    threshold: float
        threshold of new label. If abs(old_label)<threshold then new label=0 same otherwise - default value is 1.96
    """
    label_field = f"label-{name}"
    old_label_field = f"label-{old_label_name}"
    oe_label_field = "label-oe"
    ue_label_field = "label-ue"
    for comm in comm_map.values():  # for each community
        for n in comm.nodes():
            nd = comm.nodes[n]
            if label_field not in nd:
                comm.add_node(n, **{label_field: {}})
            if name == "onekernel":
                if pid in nd[oe_label_field].keys():
                    nd[label_field][pid] = nd[oe_label_field][pid]
                elif pid in nd[ue_label_field].keys():
                    nd[label_field][pid] = nd[ue_label_field][pid]
            else:
                if (
                    pid in nd[old_label_field].keys()
                    and abs(nd[old_label_field][pid]) < threshold
                ):
                    nd[label_field][pid] = 0
                elif pid in nd[old_label_field].keys():
                    nd[label_field][pid] = nd[old_label_field][pid]
