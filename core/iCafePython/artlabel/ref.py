import os
import numpy as np
import networkx as nx
import copy
from rich import print

BOITYPENUM = 23
VESTYPENUM = 25
edgefromnode = [[] for i in range(BOITYPENUM)]
#node 0 can have any edge type
edgefromnode[0] = [0]
edgefromnode[1] = [1]
edgefromnode[2] = [2]
edgefromnode[3] = [1, 3, 7]
edgefromnode[4] = [2, 4, 8]
edgefromnode[5] = [7, 9, 11]
edgefromnode[6] = [8, 10, 11]
edgefromnode[7] = [3, 5, 5]
edgefromnode[8] = [4, 6, 6]
edgefromnode[9] = [1, 1, 23]
edgefromnode[10] = [2, 2, 24]
edgefromnode[11] = [23]
edgefromnode[12] = [24]
#skip m23
edgefromnode[15] = [14]
edgefromnode[16] = [15]
edgefromnode[17] = [14, 15, 16]
edgefromnode[18] = [16, 17, 18]
edgefromnode[19] = [17, 19, 21]
edgefromnode[20] = [18, 20, 22]
edgefromnode[21] = [1, 1, 21]
edgefromnode[22] = [2, 2, 22]

nodeconnection = [[] for i in range(BOITYPENUM)]
nodeconnection[3] = [1, 5, 7]
nodeconnection[4] = [2, 6, 8]
nodeconnection[18] = [17, 19, 20]

center_node_prob_thres = 1e-10

#definition of key sets
compset = set([1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20])

#edgetype:[nodetypes]
edgemap = {11: [5, 6], 21: [19, 21], 22: [20, 22], 23: [9, 11], 24: [10, 12], 14: [15, 17], 15: [16, 17]}
#for M2 A2 P2, from proximal to distal, third number is edge type
fillmap = [[3, 7, 5], [4, 8, 6], [3, 5, 9], [4, 6, 10], [18, 19, 19], [18, 20, 20]]

#ves branch length
#branchfilename = 'branch_dist.npy'
#if os.path.exists(branchfilename):

branch_dist_mean = {
    1: 84.48811569424625,
    2: 82.90483700892038,
    3: 19.446271672499304,
    4: 19.969703494698066,
    7: 19.091299466949913,
    8: 19.798758220267878,
    11: 6.043507277653538,
    14: 29.686936079673426,
    15: 28.871557487972904,
    16: 29.07940675513334,
    17: 11.96344479073318,
    18: 10.736976954601344,
    21: 17.55030900179855,
    22: 17.773095636546078,
    23: 35.44414820605388,
    24: 34.04543106474924
}
branch_dist_std = {
    1: 18.94166574327885,
    2: 20.024045542040316,
    3: 7.809567414452621,
    4: 21.815641717507383,
    7: 3.803737256508319,
    8: 4.136273801494701,
    11: 7.209074014704321,
    14: 13.181591807153046,
    15: 12.99171284589376,
    16: 9.343755354262111,
    17: 7.765852598078354,
    18: 4.802299515890282,
    21: 12.677797977255285,
    22: 6.375358095052876,
    23: 15.750171011424323,
    24: 15.00996924972025
}
#branch_dist_mean,branch_dist_std = np.load(branchfilename,allow_pickle=True)
#else:
#print('Branch length not loaded')
# branch_dist_mean = {}
# branch_dist_std = {}
# branchdists = [[] for i in range(VESTYPENUM)]
# for gi in all_db['train'][:]:
#     G = generate_graph(gi)
#     cnode_type_to_id = {}
#     for nodei in G.nodes():
#         cnodetype = G.nodes[nodei]['boitype']
#         if cnodetype!=0:
#             cnode_type_to_id[cnodetype] = nodei
#     #print(cnode_type_to_id)
#     #edge according to node
#     for nodetypei in range(len(nodeconnection)):
#         if len(nodeconnection[nodetypei])==0:
#             continue
#         if nodetypei not in cnode_type_to_id.keys():
#             continue
#         for branchnodetypei in nodeconnection[nodetypei]:
#             if branchnodetypei not in cnode_type_to_id.keys() or nodetypei not in cnode_type_to_id.keys():
#                 continue
#             edgetype = iCafe.matchvestype(nodetypei,branchnodetypei)
#             try:
#                 sp = nx.shortest_path(G,cnode_type_to_id[nodetypei],cnode_type_to_id[branchnodetypei])
#             except nx.NetworkXNoPath:
#                 print('no path between',nodetypei,branchnodetypei,gi)
#                 continue
#             cdist = nodedist(G,cnode_type_to_id[nodetypei],cnode_type_to_id[branchnodetypei])
#             branchdists[edgetype].append(cdist)

#     #additional edges based on node types
#     for edgetype,nodetypes in edgemap.items():
#         if nodetypes[0] not in cnode_type_to_id.keys() or nodetypes[1] not in cnode_type_to_id.keys():
#             continue

#         edgetype = iCafe.matchvestype(nodetypes[0],nodetypes[1])
#         try:
#             sp = nx.shortest_path(G,cnode_type_to_id[nodetypes[0]],cnode_type_to_id[nodetypes[1]])
#         except nx.NetworkXNoPath:
#             print('no path between',nodetypes,gi)
#             continue
#         cdist = nodedist(G,cnode_type_to_id[nodetypes[0]],cnode_type_to_id[nodetypes[1]])
#         branchdists[edgetype].append(cdist)
# for vestype,dists in enumerate(branchdists):
#     if len(dists)==0:
#         continue
#     branch_dist_mean[vestype] = np.mean(dists)
#     branch_dist_std[vestype] = np.std(dists)
#     print(vestype,branch_dist_mean[vestype],branch_dist_std[vestype])


def softmax_probs(xi):
    e = np.exp(xi)
    return e / np.sum(e)


def findedgeid(graph, startnodeid, endnodeid):
    for i, edgei in enumerate(graph.edges()):
        if edgei == (startnodeid,endnodeid) or \
            edgei == (endnodeid,startnodeid):
            return i
    return -1


def nodedist(G, startnodeid, endnodeid):
    cdist = 0
    sp = nx.shortest_path(G, startnodeid, endnodeid)
    for spi in range(1, len(sp)):
        edgei = findedgeid(G, sp[spi - 1], sp[spi])
        if edgei != -1:
            cdist += G.edges()[sp[spi - 1], sp[spi]]['dist']
        else:
            print('cannot find id for edge', sp[spi - 1], sp[spi])
    return cdist


#nodeid neighbors are one deg1 and two deg 3+
def neideg3(graph, nodeid):
    neiids = [i[1] for i in list(graph.edges(nodeid))]
    if len(neiids) == 1:
        return False
    nndeg = [graph.nodes[ni]['deg'] for ni in neiids]
    print('nndeg', nndeg)
    #in case more than 3 neis, choose larger 3
    if sorted(nndeg)[-3:] == [1, 3, 3] or sorted(nndeg) == [3, 3, 3]:
        if sorted(nndeg)[-3:] == [1, 3, 3]:
            deg_min_node_id = neiids[np.argmin(nndeg)]
            edgeid = findedgeid(graph, nodeid, deg_min_node_id)
            if edgeid != -1:
                min_branch_dist = graph.edges[list(graph.edges())[edgeid]]['dist']
                if min_branch_dist < 10:
                    return False
            else:
                print('no edgeid', nodeid, deg_min_node_id)
        return True
    else:
        return False


def findmaxprob(graph, probnodes, nodeid, visited, targettype, targetdeg, vestype=0, majornode=0):
    #first node in visited is the root node
    #nodeid gives the direction to search along that branch
    validnode = {}
    validnode[nodeid] = probnodes[nodeid][targettype]
    pendingnode = [nodeid]
    while len(pendingnode):
        nodestart = pendingnode[0]
        del pendingnode[0]
        neinodes = list(graph.edges(nodestart))
        for ni in neinodes:
            if ni[1] in visited:
                continue
            visited.append(ni[1])
            if targetdeg is None or graph.nodes[ni[1]]['deg'] == targetdeg:
                validnode[ni[1]] = probnodes[ni[1]][targettype]
            if graph.nodes[ni[1]]['deg'] >= 3:
                if vestype in branch_dist_mean.keys():
                    disttoroot = nodedist(graph, ni[1], visited[0])
                    if disttoroot > branch_dist_mean[vestype] + 2 * branch_dist_std[vestype]:
                        #print('beyond thres',vestype,branch_dist_mean[vestype],branch_dist_std[vestype])
                        continue
                pendingnode.append(ni[1])
    print(validnode)
    if majornode:
        majorvalidnode = copy.copy(validnode)
        for ki in list(majorvalidnode):
            if neideg3(graph, ki) == False:
                del majorvalidnode[ki]
        print('majornode', majorvalidnode)
        if len(majorvalidnode) == 0:
            print('major node empty')
            return max(validnode, key=validnode.get)
        else:
            return max(majorvalidnode, key=majorvalidnode.get)
    else:
        return max(validnode, key=validnode.get)


#find id of edges connected to nodeid
def find_node_connection_ids(graph, nodeid):
    edges = list(graph.edges())
    nei_edge_ids = []
    neiedges = graph.edges(nodeid)
    for edgei in neiedges:
        if edgei not in edges:
            edgei = (edgei[1], edgei[0])
        edgeid = edges.index(edgei)
        nei_edge_ids.append(edgeid)
    return nei_edge_ids


def find_nei_nodes(graph, nodeid):
    nei_node_ids = []
    neiedges = graph.edges(nodeid)
    for edgei in neiedges:
        nei_node_ids.append(edgei[1])
    return nei_node_ids


def findallnei(graph, nodeid, visited, targetdeg=None):
    validnode = []
    validedge = []
    pendingnode = [nodeid]
    while len(pendingnode):
        nodestart = pendingnode[0]
        del pendingnode[0]
        neinodes = list(graph.edges(nodestart))
        for ni in neinodes:
            if ni[1] in visited:
                continue
            visited.append(ni[1])
            if targetdeg is None or graph.nodes[ni[1]]['deg'] == targetdeg:
                validnode.append(ni[1])
                validedge.append(ni)
            if graph.nodes[ni[1]]['deg'] >= 3:
                pendingnode.append(ni[1])
    return validnode, validedge


def matchbranchtype(nei_node_ids, branch_miss_type, centernodeid):
    maxprobs = 0
    maxli = None
    for li in list(permutations(nei_node_ids, len(nei_node_ids))):
        cprobs = 0
        for item in range(len(nei_node_ids)):
            nodeid = li[item]
            edgeid = findedgeid(graph, centernodeid, nodeid)
            cprob = probnodes[edgeid][branch_miss_type[item]]
            cprobs += cprob
        #print(li,cprobs)
        if cprobs > maxprobs:
            maxprobs = cprobs
            maxli = li
    #print(maxli)
    return maxli