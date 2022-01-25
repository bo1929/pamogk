#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections as coll
import json

import networkx as nx

from pamogk import config
from pamogk.lib.sutils import *


DATA_ROOT = config.DATA_DIR / "communities"
COMM_PRE = "Bigclam_HPA-PROTEIN-KIDNEY"
COMM_MAP = f"{COMM_PRE}_comm_map.json"
COMM_LIST_PATH = DATA_ROOT / COMM_MAP
safe_create_dir(DATA_ROOT)


def get_community_map():
    if not COMM_LIST_PATH.exists():
        raise ValueError("Given community list cant be found in the given path!")

    community_list = json.load(open(COMM_LIST_PATH))

    community_map = coll.OrderedDict()
    for p in community_list["communities_all"]:
        community_map[p["@id"]] = p
    return community_map


@timeit
def read_communities(comm_pre):
    global COMM_PRE 
    global COMM_MAP 
    global COMM_LIST_PATH
    COMM_PRE = comm_pre
    COMM_MAP = f"{COMM_PRE}_comm_map.json"
    COMM_LIST_PATH = DATA_ROOT / COMM_MAP
    safe_create_dir(DATA_ROOT)

    community_map = get_community_map()
    comm_map = coll.OrderedDict()
    comm_ids = community_map.keys()
    log(f"Community data_dir={DATA_ROOT}")
    for (ind, comm_id) in enumerate(comm_ids):
        log(f"Processing community {ind + 1:3}/{len(comm_ids)}", end="\t")
        comm_data = read_single_community(comm_id, reading_all=True)
        comm_map[comm_id] = comm_data


    log()
    return comm_map


def read_single_community(community_id, reading_all=False):
    pend = "\r" if reading_all else "\n"
    community_map = get_community_map()
    if community_id not in community_map.keys():
        raise Exception("Community not found in community list")

    COMMUNITY_PATH = DATA_ROOT / f"{COMM_PRE}_comm{community_id}.json"

    if not COMM_LIST_PATH.exists():
        raise ValueError("Given community list cant be found in given path!")

    else:
        log(
            f"Community with community_id={community_id} retrieved from local data dir",
            end=pend,
            ts=not reading_all,
        )

    community_data = json.load(open(COMMUNITY_PATH))

    G = nx.Graph()  # initialize empty graph

    # get node map
    nodes = community_data["community_nodes"]

    # add nodes to graph
    # NOTE networkx graphs only allow alphanumeric characters as attribute names no - or _
    for nid, ent_ids in nodes.items():
        G.add_node(int(nid), **{"type": "Protein", "entrezids": [ent_ids]})

    # get edge map
    edges = community_data["community_edges"]

    # add edges to graph
    for eid in edges.keys():
        e = edges[eid]
        ## attrs = {}
        G.add_edge(int(e["s"]), int(e["t"]))  ## , **attrs)
    return G


if __name__ == "__main__":
    pass
