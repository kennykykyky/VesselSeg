import networkx as nx
import numpy as np
from .definition import BOITYPENUM, VESTYPENUM, VesselName, NodeName
from collections import Counter
import os
import matplotlib.pyplot as plt
from .utils.graph_utils import _refreshid, _read_pickle_graph, _write_pickle_graph
from .point3d import Point3D
from .swcnode import SWCNode
from .snake import Snake
from .snakelist import SnakeList


# save .graph file in icafe result folder
def generateGraph(self, trace_src='raw_ves', ves_src='ves', graphtype='graphsim', mode='test', trim=1):
    snakelist, swclist = self.loadSWC(trace_src)
    #refresh/assign degree for each swcnode
    snakelist.assignDeg()
    ves_snakelist = None
    if graphtype == 'graphsim':
        if mode == 'train':
            ves_snakelist = self.loadVes(ves_src, mode='r')
            Gs = generateSimG(snakelist, ves_snakelist, self.xml.landmark, ASSIGNNODE=1, ASSIGNEDGE=1, ASSIGNDIR=1)
        elif mode == 'test':
            Gs = generateSimG(snakelist, ves_snakelist, self.xml.landmark, ASSIGNNODE=0, ASSIGNEDGE=0, ASSIGNDIR=1)
    elif graphtype == 'graphcom':
        return generateVesGraph(swclist)
    else:
        raise ValueError('Undefined graph type')

    if len(self.snakelist) == 0:
        print('no swc snake list')
        return None
    # self.xml.getLandmark(IGNOREM3=1)

    if trim:
        S = []
        for c in nx.connected_components(Gs):
            Gsi = Gs.subgraph(c).copy()
            gsidist = np.sum([Gs.edges[nodei]['dist'] for nodei in Gsi.edges()])
            # print(len(c),gsidist)
            if gsidist > 100 and len(Gsi.nodes()) > 5:
                S.append(Gsi)
        # sort based on length
        SSort = []
        for i in np.argsort([len(c) for c in S])[::-1]:
            SSort.append(S[i])
        if len(SSort) == 0:
            G = Gs
        else:
            G = _refreshid(nx.compose_all(SSort))
    else:
        G = Gs

    VESCOLORS = ['b'] + ['r'] * VESTYPENUM
    NODECOLORS = ['r'] + ['b'] * BOITYPENUM
    posz = {k: [-v['pos'].pos[0], -v['pos'].pos[1]] for k, v in G.nodes.items()}
    if mode == 'train':
        edgecolors = [VESCOLORS[np.argmax(G.edges[v]['vestype'])] for v in G.edges()]
        nodecolors = [NODECOLORS[G.nodes[n]['boitype']] for n in G.nodes()]
    else:
        edgecolors = [VESCOLORS[0] for v in G.edges()]
        nodecolors = [NODECOLORS[0] for n in G.nodes()]
    plt.figure(figsize=(5, 5))
    nx.draw_networkx(G, pos=posz, node_size=30, node_color=nodecolors, edge_color=edgecolors)
    plt.show(block=False)
    '''misslandmk = []
	for i, j in self.xml.landmark:
		# print(self.VesselName[i],j)
		if j.hashPos() not in self.simghash:
			# print(self.VesselName[i],'Not Found')
			misslandmk.append(NodeName[i])
	print('total landmark', len(self.xml.landmark), 'miss', len(misslandmk), misslandmk)'''

    for nodei in G.nodes():
        G.nodes[nodei]['pos'] = G.nodes[nodei]['pos'].pos
        G.nodes[nodei]['deg'] = G.degree[nodei]
        if mode == 'train':
            #remove extend types for distal branches
            if G.nodes[nodei]['boitype'] > 22:
                G.nodes[nodei]['boitype'] = 0
        elif mode == 'test':
            G.nodes[nodei]['boitype'] = 0
    if mode == 'train':
        for edgei in G.edges():
            if G.edges[edgei]['vestype'][12] > 0:
                #print('merge m23')
                G.edges[edgei]['vestype'][5] += G.edges[edgei]['vestype'][12]
                G.edges[edgei]['vestype'][12] = 0
            if G.edges[edgei]['vestype'][13] > 0:
                G.edges[edgei]['vestype'][6] += G.edges[edgei]['vestype'][13]
                G.edges[edgei]['vestype'][13] = 0
    elif mode == 'test':
        for edgei in G.edges():
            G.edges[edgei]['vestype'] = 0
    return G


def writeGraph(self, G, graphtype='graphsim', path=None):
    if path is None:
        pickle_graph_name = self.path + '/' + graphtype + '_TH_' + self.filename_solo + '.pickle'
    else:
        pickle_graph_name = path
    _write_pickle_graph(pickle_graph_name, G)


# load graph from result folder if exist
def readGraph(self, graphtype='graphsim'):
    pickle_graph_name = self.path + '/' + graphtype + '_TH_' + self.filename_solo + '.pickle'
    return _read_pickle_graph(pickle_graph_name)


def generateVesGraph(swclist):
    G = nx.Graph()
    for swci in swclist:
        G.add_node(swci.id, pos=swci.pos, rad=swci.rad, type=swci.type, pid=swci.pid)
        if swci.pid != -1:
            G.add_edge(swci.id, swci.pid, dist=G.nodes[swci.pid]['pos'].dist(swci.pos))
    return G


# construct graph with only key nodes (deg!=2)
def generateSimG(snakelist, ves_snakelist, landmark, ASSIGNNODE=0, ASSIGNEDGE=0, ASSIGNDIR=0):
    swclist = snakelist.toSWCList()
    if ASSIGNNODE:
        if len(landmark) == 0:
            raise ValueError('landmark is empty')
        landmarkposmap = {}
        for lmtype, lmpos in landmark:
            # ignore M2/3
            if lmtype in [13, 14]:
                continue
            landmarkposmap[lmpos.hashPos()] = lmtype

    if ASSIGNEDGE:
        if len(swclist) == 0:
            raise ValueError('Vessnake empty')
        #veslist [[] for i in range(VESTYPE))]
        veslist = ves_snakelist.toVesList()
        vesposmap = {}
        for ctype in range(1, VESTYPENUM):
            if len(veslist[ctype]) == 0:
                continue
            for snakei in veslist[ctype]:
                # skip start and end point for ves type mapping (might appear in multiple snake)
                for nodeid in range(1, len(snakei) - 1):
                    vesposmap[snakei[nodeid].pos.hashPos()] = ctype

    if ASSIGNDIR:
        availdirs = [Point3D(0, 0, -1), Point3D(0, 0, 1)]
        for ag1 in range(-45, 46, 45):
            for ag2 in range(0, 360, 45):
                xi = np.cos(ag1 / 180 * np.pi) * np.sin(ag2 / 180 * np.pi)
                yi = np.cos(ag1 / 180 * np.pi) * np.cos(ag2 / 180 * np.pi)
                zi = np.sin(ag1 / 180 * np.pi)
                cdir = Point3D(xi, yi, zi) # .norm()
                availdirs.append(cdir)

    startswcid = None # record the starting id of the seg, index is node id, value: first id is the swclist id, second is the node id in graph
    G = nx.Graph()
    simghash = [] # hash for all pos in simg, same id as graph id

    # first node
    cti = swclist[0]
    startswcid = 0 # swclist id
    G.add_node(len(G.nodes), swcid=cti.id, pos=cti.pos, rad=cti.rad, deg=cti.type)
    simghash.append(cti.pos.hashPos())

    rads = []
    dists = []
    for i in range(1, len(swclist)):
        cnode = swclist[i]
        prevnode = swclist[i - 1]
        rads.append(cnode.rad)
        if cnode.pid == prevnode.id:
            dists.append(prevnode.pos.dist(cnode.pos))
        else:
            dists.append(0)
        # type in raw_ves is degree
        if cnode.type != 2:
            # add node if not exist
            if cnode.pos.hashPos() not in simghash:
                cnodeGid = len(G.nodes)
                G.add_node(cnodeGid, swcid=cnode.id, pos=cnode.pos, rad=cnode.rad, deg=cnode.type)
                simghash.append(cnode.pos.hashPos())
                if ASSIGNNODE:
                    cnodehash = cnode.pos.hashPos()
                    if cnodehash in landmarkposmap:
                        lmtype = landmarkposmap[cnodehash]
                        G.add_node(cnodeGid, boitype=lmtype)
            else:
                cnodeGid = simghash.index(cnode.pos.hashPos())
            # rint('Line',i,'Node',cnode,'gid',cnodeGid)
            # add edge with feature if has at least two rads
            if cnode.pid != -1:
                assert swclist[startswcid].pos.hashPos() in simghash
                startGid = simghash.index(swclist[startswcid].pos.hashPos())
                mdswcid = (startswcid + i) // 2
                mdnode = swclist[mdswcid]
                mdnodeposhash = mdnode.pos.hashPos()
                if ASSIGNEDGE:
                    if mdnodeposhash in vesposmap:
                        edgetype = vesposmap[mdnodeposhash]
                    else:
                        # print('no ves for pt',mdnode)
                        edgetype = 0
                    edgetype_onehot = [0] * VESTYPENUM
                    edgetype_onehot[edgetype] = 1
                    G.add_edge(startGid, cnodeGid, dist=np.sum(dists), rad=np.mean(rads), vestype=edgetype_onehot)
                else:
                    G.add_edge(startGid, cnodeGid, dist=np.sum(dists), rad=np.mean(rads))
                #print(i,startswcid,'connect',startGid,cnodeGid,'len',len(rads))
                if ASSIGNDIR:
                    dirgap = 3
                    startswcidend = min(startswcid + dirgap, startswcid + len(rads) - 1)
                    # add direction to startnode
                    dirs = G.nodes[startGid].get('dir')
                    if dirs is None:
                        dirs = np.zeros(len(availdirs))
                    startdir = swclist[startswcidend].pos - swclist[startswcid].pos
                    startdirnorm = startdir / startdir.vecLenth()
                    startmatchdir = startdirnorm.posMatch(availdirs)
                    dirs[startmatchdir] += 1
                    G.add_node(startGid, dir=dirs)

                    # add direction to endnode
                    endswcidend = max(i - dirgap, i - len(rads) + 1)
                    dirs = G.nodes[cnodeGid].get('dir')
                    if dirs is None:
                        dirs = np.zeros(len(availdirs))
                    enddir = swclist[endswcidend].pos - swclist[i].pos
                    enddirnorm = enddir / enddir.vecLenth()
                    endmatchdir = enddirnorm.posMatch(availdirs)
                    dirs[endmatchdir] += 1
                    G.add_node(cnodeGid, dir=dirs)

                    # add direction to edge
                    edgedir = G.nodes[startGid]['pos'] - G.nodes[cnodeGid]['pos']
                    # print(startGid,cnodeGid,G.nodes[startGid]['pos'],G.nodes[cnodeGid]['pos'],edgedir)
                    edgedirnorm = edgedir / edgedir.vecLenth()

                    # dirs = np.zeros(len(availdirs))
                    # edgematchdir = edgedirnorm.posmatch(availdirs)
                    # dirs[edgematchdir] += 1
                    # edgedirnorm = -edgedirnorm
                    # edgematchdir = edgedirnorm.posmatch(availdirs)
                    # dirs[edgematchdir] += 1
                    # G.add_edge(startGid,cnodeGid,dir=dirs)

                    if edgedirnorm.z < 0:
                        edgedirnorm = -edgedirnorm
                    G.add_edge(startGid, cnodeGid, dir=edgedirnorm.pos)

            # new node from this node
            startswcid = i
            rads = []
            dists = []
            rads.append(cnode.rad)

    if len(landmark):
        for nodei in G.nodes():
            fd = -1
            for li, posi in landmark:
                if posi.dist(G.nodes[nodei]['pos']) == 0:
                    G.add_node(nodei, boitype=li)
                    fd = 1
                    break
            if fd == -1:
                G.add_node(nodei, boitype=0)

    return G


# generate graph using raw_ves file and setting
def generateG(self, ASSIGNNODE=1, ASSIGNEDGE=0):
    # ct.id starts from 1 in raw_ves, but graph requires id start from 0, ct.id-1 needed
    # map from point to graph id
    pointmap = {}
    node_w = []
    for i in range(len(self.swclist)):
        ct = self.swclist[i]
        node_w.append(tuple([ct.id - 1, {'swcid': ct.id, 'pos': ct.pos, 'rad': ct.rad}]))
        pointmap['-'.join(['%.3f' % i for i in ct.pos.pos])] = ct.id - 1
    G = nx.Graph()
    G.add_nodes_from(node_w)
    edge_w = []
    for i in range(len(self.swclist)):
        ct = self.swclist[i]
        if ct.id != -1 and ct.pid != -1:
            cdist = G.nodes[ct.id - 1]['pos'].dist(G.nodes[ct.pid - 1]['pos'])
            crad = (G.nodes[ct.id - 1]['rad'] + G.nodes[ct.pid - 1]['rad']) / 2
            edge_w.append(tuple([ct.id - 1, ct.pid - 1, {'dist': cdist, 'rad': crad}]))
    G.add_edges_from(edge_w)

    if ASSIGNNODE:
        if len(self.xml.landmark) == 0:
            self.getlandmark()
        if len(self.xml.landmark) == 0:
            print('no landmark assigned, skip landmarks')
            return G
        # assign landmark to node in graph
        for nodei in G.nodes():
            # default 0
            G.add_node(nodei, boitype=0)
            for lmtype, lmpos in self.xml.landmark:
                if G.nodes[nodei]['pos'].dist(lmpos) == 0:
                    G.add_node(nodei, boitype=lmtype)
                    break

    if ASSIGNEDGE:
        if len(self.vessnakelist) == 0:
            self.loadvesnochange()
        if len(self.vessnakelist) == 0:
            print('no landmark assigned, skip landmarks')
            return G
        for ctype in range(1, 25):
            if len(self.veslist[ctype]) == 0:
                continue
            for snakei in self.veslist[ctype]:
                for nodeid in range(1, len(snakei)):
                    nodei = snakei[nodeid - 1]
                    nodej = snakei[nodeid]
                    keyi = '-'.join(['%.3f' % i for i in nodei.pos.pos])
                    keyj = '-'.join(['%.3f' % i for i in nodej.pos.pos])
                    if keyi in pointmap.keys() and keyj in pointmap.keys():
                        G.add_edge(pointmap[keyi], pointmap[keyj], vestype=ctype)
                    else:
                        print(keyi, keyj, ctype)
        for edgei in G.edges(data=True):
            if 'vestype' not in G.edges[edgei[0], edgei[1]]:
                G.add_edge(edgei[0], edgei[1], vestype=0)
    return G


# generate graph using ves file
def generateGfromves(self):
    if len(self.vessnakelist) == 0:
        self.loadves()
    assert len(self.vessnakelist) > 0
    G = nx.Graph()
    for ptid, swci in enumerate(self.ptlist):
        G.add_node(swci.id, pos=swci.pos, rad=swci.rad, boitype=swci.type, deg=len(self.deglist[swci.id]))
    for vessnakei in self.vessnakelist:
        if vessnakei.NP < 2:
            print('skip length <2 ves')
            continue
        for swci in range(1, len(vessnakei.snake)):
            G.add_edge(vessnakei.snake[swci - 1].id, vessnakei.snake[swci].id, vestype=vessnakei.type,
                       rad=(vessnakei.snake[swci - 1].rad + vessnakei.snake[swci].rad) / 2)
    return G


def findSnakeFromPt(Gves, G, ves_end_pt):
    # G is the simple graph, Gves is the complete graph
    node_1, node_type_1, node_2, node_type_2, ctype = ves_end_pt
    startnid = []
    endnid = []
    for ni in Gves.nodes():
        if Gves.nodes[ni]['pos'].dist(Point3D(G.nodes[node_1]['pos'])) < 0.01:
            # print(ni,Gves.nodes[ni]['pos'])
            startnid.append(ni)
        if Gves.nodes[ni]['pos'].dist(Point3D(G.nodes[node_2]['pos'])) < 0.01:
            # print(ni,Gves.nodes[ni]['pos'])
            endnid.append(ni)
    if len(startnid) == 0 or len(endnid) == 0:
        print('not found in graph')
        return

    paths = []
    for si in startnid:
        for ei in endnid:
            if nx.has_path(Gves, si, ei):
                paths.append(nx.shortest_path(Gves, si, ei))
    path_lengths = [len(path) if path is not None else np.inf for path in paths]
    if not path_lengths:
        print('no path')
        return
    if min(path_lengths) != np.inf:
        min_snake = Snake(type=ctype)
        for p in paths[np.argmin(path_lengths)]:
            min_snake.addSWC(Gves.nodes[p]['pos'], Gves.nodes[p]['rad'])
        min_snake[0].type = node_type_1
        min_snake[-1].type = node_type_2
        return min_snake
    else:
        print('no path')
        return


def findSnakeFromPts(Gves, G, ves_end_pts):
    ves_snakelist = SnakeList()
    for ves_end_pt in ves_end_pts:
        new_snake = findSnakeFromPt(Gves, G, ves_end_pt)
        if new_snake is None:
            continue
        ves_snakelist.addSnake(new_snake)
    ves_snakelist = ves_snakelist.removeShort(3)
    return ves_snakelist