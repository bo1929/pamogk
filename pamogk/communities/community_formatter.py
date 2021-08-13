#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections as coll
import json

from .. import config
from ..lib.sutils import *

from pamogk.gene_mapper import uniprot_mapper

DATA_ROOT = config.DATA_DIR / "communities"
DETECTION_ALGORITHM = "Bigclam"
COMM_TYPE = "HPA-PROTEIN-KIDNEY"
COMM_PATH = DATA_ROOT / DETECTION_ALGORITHM / COMM_TYPE
COMM_NAME = f"{DETECTION_ALGORITHM}_{COMM_TYPE}"
safe_create_dir(DATA_ROOT)

def read_communities_from_txt():
    if not COMM_PATH.exists():
        raise ValueError(
            "Given community path does not exit!"
        )

    hg = lambda x: list(map(int, x.strip().split('\t')))

    with open(COMM_PATH / "community_list.txt" ) as f:
        content = f.readlines()
        communities = [hg(x) for x in content] 

    with open(COMM_PATH / "edges.txt" ) as f:
        content = f.readlines()
        edges = [hg(x) for x in content] 

    with open(COMM_PATH / "nodes.txt" ) as f:
        content = f.readlines()
        nodes = [hg(x) for x in content] 

    with open(COMM_PATH / "prioritization.txt" ) as f:
        content = list(f.readlines())
        misc = [x.strip().split('\t') for x in content] 
        misc_attr = coll.defaultdict(list)
        for i in range(1, len(misc)):
            tmp_dict = {}
            for j, key in enumerate(misc[0]):
                tmp_dict[key] = misc[i][j]
            misc_attr[i] = tmp_dict
    comm_all = {"communities":communities, "nodes":nodes, "edges":edges, "attr":misc_attr}
    return comm_all

def wrt_communities_to_json(comm_all):
    comm_map = coll.defaultdict(list)
    st_edge_map = coll.defaultdict(list)
    ## ts_edge_map = defaultdict(list)
    id_map = {}

    for idx, node in  enumerate(comm_all["nodes"]):
        comm_map["node_all"].append({'@id':idx, 'r':node})
        id_map[node[0]] = int(idx)

    for idx, edge in  enumerate(comm_all["edges"]):
        assert len(edge) == 2
        id_0 = id_map[edge[0]]
        id_1 = id_map[edge[1]]
        comm_map["edges_all"].append({'@id':idx, 's':id_0, 't':id_1})
        st_edge_map[id_0].append({'s':id_0, 't':id_1, '@id': idx})
        ## ts_edge_map[id_1].append({'t':id_1, 's':id_0, '@id': idx)

    comm_attr = comm_all["attr"]
    for idx, community in enumerate(comm_all["communities"]):
        comm_data = coll.defaultdict(list)
        comm_data["community_nodes"] =  {
            id_map[community[i]] : community[i] for i in range(len(community))
        }
        nodes = comm_data["community_nodes"].keys()
        comm_data["community_edges"] = {edge['@id']: {'s':n, 't':edge['t']} for n in nodes for edge in st_edge_map[n] if edge['t'] in nodes}

        with open(DATA_ROOT/f"{COMM_NAME}_comm{str(idx)}.json", 'w') as f:
            json.dump(comm_data, f)

        comm_map["communities_all"].append({'@id':idx, 'name':f"comm{str(idx)}"}) ## , "attr": comm_attr[idx]})

    with open(DATA_ROOT/f"{COMM_NAME}_comm_map.json", 'w') as f:
        json.dump(comm_map, f)


if __name__=="__main__":
    wrt_communities_to_json(read_communities_from_txt())
