import numpy as np
import networkx as nx
import os
from ..definition import BOITYPENUM,VESTYPENUM,VesselName,NodeName


def _refreshid(G):
    #replace node id with new id
    Gnew = nx.Graph()
    nodemap = {}
    for newid,node in enumerate(G.nodes(data=True)):
        #print(newid,node[0])
        nodemap[node[0]] = newid
        kwargs = node[1]
        Gnew.add_node(newid,**kwargs)
    for edge in G.edges(data=True):
        #print(edge)
        kwargs = edge[2]
        Gnew.add_edge(nodemap[edge[0]],nodemap[edge[1]],**kwargs)
    return Gnew


# load graph from result folder
def _read_pickle_graph(picklegraphname):
    if not os.path.exists(picklegraphname):
        print('No graph at icafe result folder')
        return None
    return nx.read_gpickle(picklegraphname)

def _write_pickle_graph(picklegraphname,G):
    nx.write_gpickle(G, picklegraphname)
    print('Graph saved', picklegraphname, 'Node', len(G.nodes), 'Edges', len(G.edges))
