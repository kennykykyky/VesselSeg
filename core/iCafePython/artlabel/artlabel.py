import copy
from itertools import permutations
import datetime
import os
from rich import print

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models

import networkx as nx
import numpy as np
import tensorflow as tf
from ._utils import create_feed_dict, create_placeholders, make_all_runnable_in_session, create_loss_ops, \
    read_pickle_graph
from .ref import edgemap, fillmap, center_node_prob_thres, branch_dist_mean, branch_dist_std, compset, nodeconnection, \
    edgefromnode, BOITYPENUM, VESTYPENUM
from .ref import softmax_probs, findedgeid, nodedist, neideg3, findmaxprob, find_node_connection_ids, find_nei_nodes, \
    findallnei, matchbranchtype
from ..definition import matchvestype
from ..point3d import Point3D
from ..swcnode import SWCNode
from ..snake import Snake

class ArtLabel:
    def __init__(self):
        self._landmark = None
        self._veslist = None # list, first of vessel type, then each snake in that type
        self._vessnakelist = None  # everything in one Snakelist
        self._ves_end_pts = None
        self._sess = None

    @property
    def landmark(self):
        if self._landmark is None:
            self.pred()
        return self._landmark

    @property
    def veslist(self):
        if self._veslist is None:
            self.pred()
        return self._veslist

    @property
    def vessnakelist(self):
        if self._vessnakelist is None:
            self.pred()
        return self._vessnakelist

    def loadGraphName(self, pickle_graph_name, res, mean_off_LR=None):
        self.pickle_graph_name = pickle_graph_name
        self.all_db = {}
        self.all_db['train'] = [pickle_graph_name]
        self.all_db['val'] = [pickle_graph_name]
        self.all_db['test'] = [pickle_graph_name]
        self.all_db['res'] = res
        self.all_db['mean_off_LR'] = mean_off_LR

    def loadmodel(self, modelfilename=None):
        if modelfilename is None:
            modelfilename = '/mnt/desktop2/Ftensorflow/LiChen/Models/ArtLabel/ArtLabel6-10-2/model42736-0.3703-0.9658-0.8317.ckpt'
        if not os.path.exists(modelfilename + '.index'):
            print('model not exist', modelfilename)
            return
        tf.compat.v1.reset_default_graph()
        BOITYPENUM = 23
        VESTYPENUM = 25
        seed = 2
        self.rand = np.random.RandomState(seed=seed)

        # Model parameters.
        # Number of processing (message-passing) steps.
        num_processing_steps_ge = 10

        # Data / training parameters.
        batch_size_tr = 1

        # Data.
        # Input and target placeholders.
        self.input_ph, self.target_ph = create_placeholders(self.all_db, self.rand, batch_size_tr)

        # Connect the data to the model.
        # Instantiate the model.
        model = models.EncodeProcessDecode(edge_output_size=VESTYPENUM, node_output_size=BOITYPENUM)
        # A list of outputs, one per processing step.
        self.output_ops_ge = model(self.input_ph, num_processing_steps_ge)

        # Test/generalization loss.
        # loss_ops_ge = create_loss_ops(self.target_ph, self.output_ops_ge,weighted=False)
        # self.loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.

        # Lets an iterable of TF graphs be output from a session as NP graphs.
        self.input_ph, self.target_ph = make_all_runnable_in_session(self.input_ph, self.target_ph)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self._sess, modelfilename)
        print("Model restored", modelfilename)

    def pred(self, pickle_graph_name=None, res=0.4, hrref=True, mean_off_LR=None):
        #res: resolution mm/px
        #hrref: whether using hierarchical refinement after gnn prediction
        #mean_off_LR: center of ICA/MCA/ACA_L and R. Default value is based on normal imaging protocal, if your image is very different in scan location, please specify manually
        if pickle_graph_name is not None:
            self.loadGraphName(pickle_graph_name,res,mean_off_LR)
        if self._sess is None:
            self.loadmodel()
        if self._sess is None:
            print('Model load fail, abort predict')
            return

        feed_dict, raw_graphs, selids = create_feed_dict(self.all_db, self.rand, len(self.all_db['test']),
                                                         self.input_ph, self.target_ph, 'test')
        # Test/generalization loss.

        test_values = self._sess.run({
            "target": self.target_ph,
            # "loss": self.loss_op_ge,
            "outputs": self.output_ops_ge
        },
            feed_dict=feed_dict)
        targets = utils_np.graphs_tuple_to_data_dicts(test_values["target"])
        outputs = list(
            zip(*(utils_np.graphs_tuple_to_data_dicts(test_values["outputs"][i])
                  for i in range(len(test_values["outputs"])))))
        # print(outputs[0][0]['nodes'])
        self.test_values = test_values
        # direct output from GNN
        # modify output from network
        rawoutputs = []
        for ti, outputr in enumerate(outputs):
            output = copy.copy(outputr[-1])
            rawoutputs.append(copy.deepcopy(output))

        rawoutputs = copy.deepcopy(outputs)

        if hrref:
            refoutput = self.HR(rawoutputs, raw_graphs)
        else:
            refoutput = rawoutputs[0]

        self.refoutput = refoutput

        graph = read_pickle_graph(self.all_db['test'][0])
        # gather non-zero node predictions
        self._landmark = []
        for ni in range(len(refoutput['nodes'])):
            ctype = np.argmax(refoutput['nodes'][ni])
            if ctype != 0:
                clocation = Point3D(graph.nodes[ni]['pos'])
                self._landmark.append([ctype, clocation])

        #self._veslist = [[] for i in range(VESTYPENUM)]  # list, first of vessel type, then each snake in that type
        #self._vessnakelist = SnakeList()  # everything in one Snakelist
        self._ves_end_pts = []
        for ei in range(len(refoutput['edges'])):
            ctype = np.argmax(refoutput['edges'][ei])
            if ctype != 0:
                li = list(graph.edges())[ei]
                edge_id = [li[0],np.argmax(refoutput['nodes'][li[0]]),li[1],np.argmax(refoutput['nodes'][li[1]]),ctype]
                edge_id = self.checkInvalidVesID(edge_id)
                self._ves_end_pts.append(edge_id)

        return self._landmark, self._ves_end_pts

    def checkInvalidVesID(self,ves_end_pt):
        ves_typical = {}
        ves_typical[1] = (1, 3)
        ves_typical[2] = (2, 4)
        ves_typical[3] = (3, 7)
        ves_typical[4] = (4, 8)
        ves_typical[5] = (7, 13)
        ves_typical[6] = (8, 14)
        ves_typical[7] = (3, 5)
        ves_typical[8] = (4, 6)
        ves_typical[9] = (5, 23)
        ves_typical[10] = (6, 24)
        ves_typical[11] = (5, 6)
        ves_typical[12] = (13, 25)
        ves_typical[13] = (14, 26)
        ves_typical[14] = (15, 17)
        ves_typical[15] = (16, 17)
        ves_typical[16] = (17, 18)
        ves_typical[17] = (18, 19)
        ves_typical[18] = (18, 20)
        ves_typical[19] = (19, 27)
        ves_typical[20] = (20, 28)
        ves_typical[21] = (19, 21)
        ves_typical[22] = (20, 22)
        ves_typical[23] = (9, 11)
        ves_typical[24] = (10, 12)

        if matchvestype(ves_end_pt[1], ves_end_pt[3]) != ves_end_pt[-1]:
            ves_end_pt[1] = ves_typical[ves_end_pt[-1]][0]
            ves_end_pt[3] = ves_typical[ves_end_pt[-1]][1]
        return ves_end_pt

    def HR(self, outputs, raw_graphs):
        print(branch_dist_mean)
        # node matching edges

        # refinement using HR
        targets = [outputs[0][-1]]

        # test confident node acc
        fitacc = 0
        errfitacc = 0
        errlist = []

        # refined output
        refoutputs = []

        starttime = datetime.datetime.now()
        for ti in [0]:
            graph = raw_graphs[ti]
            target = targets[ti]
            output = outputs[ti]
            # find vestype in degree 2
            deg2edge = []
            deg2node = []
            # graphname = os.path.basename(all_db['test'][selids[ti]])[:-7]
            # print('='*40,'\n','ti',ti,graphname)
            # prob for all edges and nodes
            probedges = softmax_probs(output[-1]["edges"])
            probnodes = softmax_probs(output[-1]["nodes"])
            prededges = np.argmax(output[-1]["edges"], axis=1)
            prednodes = np.argmax(output[-1]["nodes"], axis=1)
            targetnodes = np.argmax(target['nodes'], axis=1)

            edges = list(graph.edges())

            # key:nodeid, value:landmarkid
            node_type_to_id = {}
            prob_conf_type = {}

            coredge = [[] for i in range(output[-1]["edges"].shape[0])]
            for nodei in graph.nodes():
                # probability of each edge
                probedge = {}
                # predicted as edge type
                prededge = []
                # neighboring edge ids
                nei_edge_ids = find_node_connection_ids(graph, nodei)

                for edgeid in nei_edge_ids:
                    probedge[edgeid] = np.max(probedges[edgeid])
                    prededge.append(prededges[edgeid])

                # node prob
                probnode = np.max(probnodes[nodei])
                prednode = prednodes[nodei]
                # print('node predict',np.argmax(probnodes[nodei]))

                # conf nodes predicted in [0,9,10,11,12,21,22] are not useful in hierachical framework
                if prednode not in [0] and sorted(prededge) == edgefromnode[prednode]:
                    if prednode not in [0, 9, 10, 11, 12, 15, 16, 21, 22]:
                        if prednode in node_type_to_id.keys():
                            print('pred node type exist', probnode, 'exist prob', prob_conf_type[prednode])
                            if probnode < prob_conf_type[prednode]:
                                print('skip')
                                continue
                            else:
                                print('override')
                        node_type_to_id[prednode] = nodei
                        prob_conf_type[prednode] = probnode

                    print('Fit edge set', prededge, 'node', prednode, 'gt', targetnodes[nodei])
                    if targetnodes[nodei] != 0:
                        fitacc += 1

                    if prednode != targetnodes[nodei]:
                        if targetnodes[nodei] != 0:
                            errfitacc += 1
                            errlist.append([prededge, prednode, targetnodes[nodei]])

            # from nodetype to nodeid
            node_id_to_type = {node_type_to_id[i]: i for i in node_type_to_id}
            print('key node unconfident', compset - set(node_id_to_type.values()))
            ##########################################
            # start with key nodes
            # anterior L(3) R(4)/posterior(18) circulation
            treenodes = {3: [1, 5, 7], 4: [2, 6, 8], 18: [17, 19, 20]}
            for center_node_type in [3, 4, 18]:
                failcenter = 0
                print('@@@Predicting centernode', center_node_type)
                # if center node not exist, find from neighbor first
                if center_node_type not in node_id_to_type.values():
                    # from confident branch nodes find center nodes
                    cofnodes = [i for i in node_id_to_type.values() if i in nodeconnection[center_node_type]]
                    print('center node not confident, branch confident nodes', cofnodes)
                    if len(cofnodes) == 0:
                        print('whole circulation cannot be confidently predicted, use max prob')
                        failcenter = 1
                    elif len(cofnodes) == 1:
                        print('predict using single node')
                        cedgetype = matchvestype(cofnodes[0], center_node_type)
                        nei_edge_ids = list(graph.edges(node_type_to_id[cofnodes[0]]))
                        nodestart = [edgei[1] for edgei in nei_edge_ids if
                                     prededges[findedgeid(graph, edgei[0], edgei[1])] == cedgetype]
                        assert len(nodestart) == 1
                        otherbranchnodes = set(
                            np.concatenate(list(treenodes.values()) + [list(treenodes.keys())])) - set(
                            treenodes[center_node_type])
                        visited = [node_type_to_id[cofnodes[0]], nodestart[0]] + [node_type_to_id[i] for i in
                                                                                  otherbranchnodes if
                                                                                  i in node_type_to_id.keys()]
                        print('exclude other branch nodeid',
                              [[i, node_type_to_id[i]] for i in otherbranchnodes if i in node_type_to_id.keys()])
                        exp_edge_type = matchvestype(cofnodes[0], center_node_type)
                        # neighbors of major nodes have degree of 1(branch length > 10),3,3 or 3,3,3
                        center_node_pred = findmaxprob(graph, probnodes, nodestart[0], visited, center_node_type, \
                                                       len(edgefromnode[center_node_type]), exp_edge_type,
                                                       majornode=True)
                    elif len(cofnodes) == 2:
                        print('predict using two nodes')
                        try:
                            # 1:-1 remove start and end node
                            pathnodeid = nx.shortest_path(graph, node_type_to_id[cofnodes[0]],
                                                          node_type_to_id[cofnodes[1]])[1:-1]
                            center_node_pred = pathnodeid[
                                np.argmax([probnodes[ni][center_node_type] for ni in pathnodeid])]
                        except nx.NetworkXNoPath:
                            print('no shortest path  between confident nodes, skip')
                            failcenter = 1

                    elif len(cofnodes) == 3:
                        print('predict using three nodes')
                        center_node_pred_set = list(
                            set(nx.shortest_path(graph, node_type_to_id[cofnodes[0]], node_type_to_id[cofnodes[1]])) & \
                            set(nx.shortest_path(graph, node_type_to_id[cofnodes[0]], node_type_to_id[cofnodes[2]])) & \
                            set(nx.shortest_path(graph, node_type_to_id[cofnodes[1]], node_type_to_id[cofnodes[2]])))
                        if len(center_node_pred_set) == 1:
                            center_node_pred = center_node_pred_set[0]
                        elif len(center_node_pred_set) == 0:
                            print('no common node for three path')
                            failcenter = 1
                        else:
                            print('center_node_pred_set has more than 1 node', center_node_pred_set)
                    else:
                        print('more than three nodes.possible?')

                    low_prob_center = 0
                    if failcenter:
                        # center_node_pred = np.argmax([probnodes[i][center_node_type] for i in range(probnodes.shape[0])])
                        center_probs = {i: probnodes[i][center_node_type] for i in range(probnodes.shape[0]) if
                                        i not in list(node_id_to_type.keys())}
                        '''
						#restrict PCA/BA location based on ICA/M1/A1 x position
						if center_node_type==18:
							#restrict left boundary
							xmax = 1
							if 3 in node_type_to_id.keys():
								xmax = graph.nodes[node_type_to_id[3]]['pos'][0]*0.8
								print('xmax',xmax)
							#restrict right boundary  
							xmin = -1
							if 4 in node_type_to_id.keys():
								xmin = graph.nodes[node_type_to_id[4]]['pos'][0]*0.8
								print('xmin',xmin)
							center_probs_minmax = {}
							for k,v in center_probs.items():
								if graph.nodes[k]['pos'][0]<xmax and graph.nodes[k]['pos'][0]>xmin:
									center_probs_minmax[k] = v
							print('after x restriction',xmin,xmax,'len',len(center_probs_minmax))
							if len(center_probs_minmax)==0:
								low_prob_center = 1
							else:
								center_node_pred = sorted(center_probs_minmax.items(), key=lambda item: -item[1])[0][0]
						else:'''
                        center_node_pred = sorted(center_probs.items(), key=lambda item: -item[1])[0][0]
                        print('center predict by most prob')

                    if probnodes[center_node_pred][center_node_type] < center_node_prob_thres or low_prob_center:
                        print('center node prob too low', probnodes[center_node_pred][center_node_type],
                              ', this may because of missing such center node type. use default')
                        # need to ensure branch nodes on ICA-M1, BA-P1 are oredicted
                        # start from root nodes
                        branch_node_type = treenodes[center_node_type][0]
                        if branch_node_type not in node_type_to_id.keys():
                            print('Use max prob to predict root node type', branch_node_type)
                            if center_node_type in [3, 4]:
                                branch_probs = {i: probnodes[i][branch_node_type] for i in range(probnodes.shape[0]) if
                                                graph.nodes[i]['deg'] == 1 and i not in list(node_id_to_type.keys())}
                            else:
                                branch_probs = {i: probnodes[i][branch_node_type] for i in range(probnodes.shape[0]) if
                                                i not in list(node_id_to_type.keys())}
                            branch_node_pred = sorted(branch_probs.items(), key=lambda item: -item[1])[0][0]
                            if probnodes[branch_node_pred][branch_node_type] < center_node_prob_thres:
                                print('no prob over thres for branch node type', branch_node_type)
                            else:
                                print('Branch node type', branch_node_type, 'nodeid', branch_node_pred, 'gt',
                                      targetnodes[branch_node_pred])
                                node_type_to_id[branch_node_type] = branch_node_pred
                                node_id_to_type[branch_node_pred] = branch_node_type
                        # A1 can be missing, M1 cannot
                        if center_node_type in [3, 4]:
                            branch_node_type2 = treenodes[center_node_type][1]
                            if branch_node_type2 not in node_type_to_id.keys():
                                print('Use max prob to predict root node type', branch_node_type2)
                                if branch_node_type in node_type_to_id.keys():
                                    visited = [node_type_to_id[branch_node_type]] + list(node_type_to_id.keys())
                                    print('search neighbors of branch nodeid', node_type_to_id[branch_node_type])
                                    nodestart = [i[1] for i in list(graph.edges(node_type_to_id[branch_node_type]))]
                                    assert len(nodestart) == 1
                                    branch_node_pred2 = findmaxprob(graph, probnodes, nodestart[0], visited,
                                                                    branch_node_type2, \
                                                                    len(edgefromnode[branch_node_type2]))
                                else:
                                    print('Root Branch not confident, use max prob for node type', branch_node_type2)
                                    branch_probs2 = {i: probnodes[i][branch_node_type2] for i in
                                                     range(probnodes.shape[0]) if
                                                     graph.nodes[i]['deg'] == 3 and i not in list(
                                                         node_id_to_type.keys())}
                                    branch_node_pred2 = sorted(branch_probs2.items(), key=lambda item: -item[1])[0][0]

                                print('Minor Branch node type', branch_node_type2, 'nodeid', branch_node_pred2, 'gt',
                                      targetnodes[branch_node_pred2])
                                node_type_to_id[branch_node_type2] = branch_node_pred2
                                node_id_to_type[branch_node_pred2] = branch_node_type2

                                # tentative A1
                                branch_node_type3 = treenodes[center_node_type][2]
                                if branch_node_type3 not in node_type_to_id.keys():
                                    print('Pred A1')
                                    visited = list(node_type_to_id.keys())
                                    print('search neighbors of branch nodeid', node_type_to_id[branch_node_type2])

                                    branch_node_pred2 = findmaxprob(graph, probnodes,
                                                                    node_type_to_id[branch_node_type2], visited, \
                                                                    branch_node_type3, \
                                                                    len(edgefromnode[branch_node_type3]))
                                    print('prob a1/2', probnodes[branch_node_pred2][branch_node_type2])
                                    if probnodes[branch_node_pred2][branch_node_type2] < center_node_prob_thres:
                                        print('no prob over thres for branch node type', branch_node_type2)
                                    else:
                                        print('Branch node type', branch_node_type2, 'nodeid', branch_node_pred2, 'gt',
                                              targetnodes[branch_node_pred2])
                                        node_type_to_id[branch_node_type2] = branch_node_pred2
                                        node_id_to_type[branch_node_pred2] = branch_node_type2
                                        # add center node back
                                        try:
                                            if branch_node_type3 in node_type_to_id:
                                                spam = nx.shortest_path(graph, node_type_to_id[branch_node_type2],
                                                                        node_type_to_id[branch_node_type3])[1:-1]
                                                if center_node_pred in spam:
                                                    print('center node in sp between A1 M1, confirm type',
                                                          center_node_type, 'nodeid', center_node_pred, 'gt',
                                                          targetnodes[center_node_pred])
                                                    node_type_to_id[center_node_type] = center_node_pred
                                                    node_id_to_type[center_node_pred] = center_node_type
                                        except nx.NetworkXNoPath:
                                            print('no shortest path between A1 M1 nodes, skip')

                        if center_node_type in [3, 4]:
                            # if ICA root exist, ICA search deg2
                            if center_node_type + 4 in node_type_to_id.keys() and center_node_type - 2 not in node_type_to_id.keys():
                                # M1 edge is center_node_type
                                deg2edge.append(center_node_type)
                            else:
                                # ICA edge is center_node_type-2
                                deg2edge.append(center_node_type - 2)
                            deg2node.append(center_node_type)
                        elif center_node_type == 18:
                            # BA search deg2
                            deg2edge.append(16)

                        deg2node.append(center_node_type)
                        continue
                    else:
                        print('pred centernode as nodeid', center_node_pred, 'prob',
                              probnodes[center_node_pred][center_node_type], 'gt', targetnodes[center_node_pred])
                        node_type_to_id[center_node_type] = center_node_pred
                        node_id_to_type[center_node_pred] = center_node_type
                else:
                    print('Centernode confident')

                nei_edges = find_node_connection_ids(graph, node_type_to_id[center_node_type])

                branch_node_types = nodeconnection[center_node_type]
                print('branch_node_types', branch_node_types)
                nei_node_ids = [i[1] for i in graph.edges(node_type_to_id[center_node_type])]
                print('nei_node_ids', nei_node_ids)
                branch_match_type = list(set(branch_node_types) & set(node_id_to_type.values()))
                branch_miss_type = list(set(branch_node_types) - set(node_id_to_type.values()))
                print('confident branch node type', branch_match_type, 'miss', branch_miss_type)
                # remove match branch nodeid
                for bi in branch_match_type:
                    if nx.has_path(graph, node_type_to_id[center_node_type], node_type_to_id[bi]):
                        sp = nx.shortest_path(graph, node_type_to_id[center_node_type], node_type_to_id[bi])
                        print('path for type', bi, sp)
                    for neini in nei_node_ids:
                        if neini in sp:
                            del nei_node_ids[nei_node_ids.index(neini)]
                    # print('remove node id',neini)
                print('nei node id remaining', nei_node_ids)
                if len(nei_node_ids) != len(branch_miss_type):
                    print('nei_node_ids', nei_node_ids, 'not missing mactch branch type', branch_miss_type)
                    if len(nei_node_ids) < len(branch_miss_type):
                        print('ERR, len(nei_node_ids)<len(branch_miss_type), abort')
                        continue

                if len(nei_node_ids):
                    # finding node from remaining branch
                    # matchnodeids = matchbranchtype(nei_node_ids,branch_miss_type,node_type_to_id[center_node_type])
                    matchnodeidsperms = list(permutations(nei_node_ids, len(branch_miss_type)))
                    maxbranchmatch = 0
                    maxbranchnodeid = None
                    for matchnodeids in matchnodeidsperms:
                        print('--Test branch case', matchnodeids)
                        cbranchmatch = 0
                        cbranchnodeid = []
                        for bi, branchtypei in enumerate(branch_miss_type):
                            # find best matching nei node id
                            nodestart = matchnodeids[bi]
                            print('branchtypei', branchtypei, 'max match node id ', nodestart)
                            visited = [node_type_to_id[center_node_type], nodestart] + list(node_type_to_id.values())
                            exp_edge_type = matchvestype(branchtypei, center_node_type)
                            # print('visited nodes id',visited,'exp ves type',exp_edge_type,'exp deg',len(edgefromnode[branchtypei]))
                            # ba/va can be either 1 or 3 in degree
                            if branchtypei == 17:
                                tnodeid = findmaxprob(graph, probnodes, nodestart, visited, branchtypei, None,
                                                      exp_edge_type)
                            else:
                                tnodeid = findmaxprob(graph, probnodes, nodestart, visited, branchtypei,
                                                      len(edgefromnode[branchtypei]), exp_edge_type)
                            cbranchmatch += probnodes[tnodeid][branchtypei]
                            cbranchnodeid.append(tnodeid)
                            print('pred branch', branchtypei, 'nodeid', tnodeid, 'prob', cbranchmatch)
                        # restrict certain type matches

                        if cbranchmatch > maxbranchmatch:
                            maxbranchmatch = cbranchmatch
                            maxbranchnodeid = cbranchnodeid
                    for bi in range(len(maxbranchnodeid)):
                        tnodeid = maxbranchnodeid[bi]
                        branchtypei = branch_miss_type[bi]
                        # check A1/2 P1/2 deg2
                        if branchtypei in [5, 6, 19, 20]:
                            cdist = 0
                            sp = nx.shortest_path(graph, node_type_to_id[center_node_type], tnodeid)
                            for spi in range(1, len(sp)):
                                cdist += graph.edges()[sp[spi - 1], sp[spi]]['dist']
                            edgetype = matchvestype(branchtypei, center_node_type)
                            if cdist > branch_dist_mean[edgetype] + 1.5 * branch_dist_std[edgetype] or \
                                    cdist < branch_dist_mean[edgetype] - 1.5 * branch_dist_std[edgetype]:
                                print(tnodeid, 'branch dist', cdist, 'over thres', branch_dist_mean[edgetype],
                                      branch_dist_std[edgetype])
                                print('add deg2 edge', edgetype, 'node', branchtypei)
                                deg2edge.append(edgetype)
                                deg2node.append(branchtypei)
                                tnodeid = sp[1]
                                print('tnodeid change to nearest', tnodeid)
                        node_id_to_type[tnodeid] = branchtypei
                        node_type_to_id[branchtypei] = tnodeid
                        print('###Best pred branch', branchtypei, 'nodeid', tnodeid, 'gt', targetnodes[tnodeid])

            # check p1/2 left right
            if 19 in node_type_to_id.keys() and 20 in node_type_to_id:
                op12lnodeid = node_type_to_id[19]
                op12rnodeid = node_type_to_id[20]
                p12lnodex = graph.nodes[op12lnodeid]['pos'][0]
                p12rnodex = graph.nodes[op12rnodeid]['pos'][0]
                # left is larger in icafe axis, if p12L x cordinate is smaller, means error
                if p12lnodex < p12rnodex:
                    print('P12LR swap')
                    node_type_to_id[19] = op12rnodeid
                    node_type_to_id[20] = op12lnodeid
                    node_id_to_type[op12lnodeid] = 20
                    node_id_to_type[op12rnodeid] = 19

            # check major branch between M1 ICA/M1/A1, and A1 ICA/M1/A1
            for mtype in [5, 6, 7, 8]:
                if mtype in [5, 6]:
                    # ICA/M1/A1 -2
                    icamatype = mtype - 2
                if mtype in [7, 8]:
                    # ICA/M1/A1 -4
                    icamatype = mtype - 4
                if icamatype not in node_type_to_id.keys() or mtype not in node_type_to_id.keys():
                    continue
                if nx.has_path(graph, node_type_to_id[icamatype], node_type_to_id[mtype]):
                    spm1 = nx.shortest_path(graph, node_type_to_id[icamatype], node_type_to_id[mtype])
                else:
                    spm1 = []
                if len(spm1) > 2:
                    print('AM12', mtype, 'nodes between AM1 ICA/M1/A1', len(spm1))
                    # current m2 nodes, spm1[-2] is closest node to m1/2
                    keyids = [nodeids for nodeids in node_type_to_id.values()]
                    fillnodes, filledges = findallnei(graph, node_type_to_id[mtype], [spm1[-2]] + keyids)
                    print('current M2 # nodes', len(fillnodes))
                    # search nodes between M1/2 and ICA/M1/A1 with most neighbor nodes
                    bestm1 = node_type_to_id[mtype]
                    nodesm2 = 0
                    for pm1 in spm1[1:-1]:
                        fillnodes, filledges = findallnei(graph, pm1, spm1 + keyids)
                        print(pm1, 'Potential AM1 in between has ', len(fillnodes), 'nodes')
                        if len(fillnodes) > 4:
                            bestm1 = pm1
                            nodesm2 = len(fillnodes)
                            break
                    if nodesm2 > 4:
                        print('AM2L node', bestm1, 'is a major branch, node', bestm1, 'set to', mtype)
                        # replace current m1/2
                        del node_id_to_type[node_type_to_id[mtype]]
                        node_type_to_id[mtype] = bestm1
                        node_id_to_type[bestm1] = mtype

            remain_keynodes = list(compset - set(node_type_to_id.keys()))
            print('key set remaining', remain_keynodes)

            ##########################################
            # additional node connection based on main tree
            # acomm check
            if 5 in node_type_to_id.keys() and 6 in node_type_to_id.keys():
                try:
                    sp = nx.shortest_path(graph, node_type_to_id[5], node_type_to_id[6])
                    if len(sp) >= 2:
                        for spi in sp[1:-1]:
                            if spi in node_type_to_id.values():
                                print('shortest acomm path come through confident nodes, abort')
                                deg2edge.append(11)
                                break
                    else:
                        deg2edge.append(11)
                        print('len between a12lr is', len(sp))
                    if 11 not in deg2edge:
                        acommdist = nodedist(graph, node_type_to_id[5], node_type_to_id[6])
                        print('acomm dist', acommdist)
                        if acommdist > branch_dist_mean[11] + 3 * branch_dist_std[11]:
                            print('likely to have overlap in A2LR')
                            deg2edge.append(11)
                except nx.NetworkXNoPath:
                    print('5,6 not connected, deg2 search needed fo edge 11 and node 5, 6')
                    deg2edge.append(11)

                if 11 not in deg2edge:
                    if 5 in deg2node:
                        del deg2node[deg2node.index(5)]
                        del deg2edge[deg2edge.index(7)]
                        print('acomm has path, remove deg 2 search for node 5')
                    if 6 in deg2node:
                        del deg2node[deg2node.index(6)]
                        del deg2edge[deg2edge.index(8)]
                        print('acomm has path, remove deg 2 search for node 6')
                    spa1l = nx.shortest_path(graph, node_type_to_id[5], node_type_to_id[3])
                    spa1r = nx.shortest_path(graph, node_type_to_id[6], node_type_to_id[4])
                    # check common path between acomm and a1
                    if len(sp) > 2:
                        for spi in sp[1:-1]:
                            # in case a1/2 is more distal than expected
                            if spi in spa1l:
                                print('set a1L/acomm to last common path', spi)
                                node_type_to_id[5] = spi
                                node_id_to_type[spi] = 5
                            if spi in spa1r:
                                print('set a1R/acomm to last common path', spi)
                                node_type_to_id[6] = spi
                                node_id_to_type[spi] = 6

            # Check OA
            for oanodetype in [9, 10]:
                if oanodetype == 9:
                    if 1 not in node_type_to_id.keys() or 3 not in node_type_to_id.keys():
                        continue
                    sp = nx.shortest_path(graph, node_type_to_id[1], node_type_to_id[3])
                elif oanodetype == 10:
                    if 2 not in node_type_to_id.keys() or 4 not in node_type_to_id.keys():
                        continue
                    if nx.has_path(graph, node_type_to_id[2], node_type_to_id[4]):
                        sp = nx.shortest_path(graph, node_type_to_id[2], node_type_to_id[4])
                    else:
                        sp = []
                else:
                    print('no such oanodetype')
                    sp = []

                maxprob = 0
                oaid = -1
                if len(sp) > 2:
                    for spid in range(1, len(sp) - 1):
                        spi = sp[spid]
                        # find oa from nei of oa/ica
                        neinodes = list(set(find_nei_nodes(graph, spi)) - set([sp[spid - 1], sp[spid + 1]]))
                        if len(neinodes) == 0:
                            print('Self loop', spi, 'ERR NEED CORR')
                            continue
                        if len(neinodes) > 1:
                            print('node 3+ at oa/ica', neinodes)
                        # oa to the front
                        print(spi, 'oa/ica search deg', graph.nodes[neinodes[0]]['deg'],
                              graph.nodes[neinodes[0]]['pos'], graph.nodes[spi]['pos'])
                        if graph.nodes[neinodes[0]]['deg'] == 1 and graph.nodes[neinodes[0]]['pos'][1] < \
                                graph.nodes[spi]['pos'][1]:
                            # if prednodes[spi]==oanodetype:
                            # print('oa/ica prob',probnodes[spi][oanodetype])
                            if probnodes[spi][oanodetype] > maxprob:
                                # print('max',spi)
                                maxprob = probnodes[spi][oanodetype]
                                oaid = spi
                    if oaid != -1:
                        print(oanodetype, 'oai/ica node', oaid, 'gt', targetnodes[oaid])
                        node_type_to_id[oanodetype] = oaid
                        node_id_to_type[oaid] = oanodetype

                        # find ending node of oa
                        neinodeids = [i[1] for i in graph.edges(oaid) if i[1] not in sp]
                        #assert len(neinodeids) == 1
                        visited = [oaid, neinodeids[0]]
                        exp_edge_type = matchvestype(oanodetype, oanodetype + 2)
                        # oa end id +2
                        oaendid = findmaxprob(graph, probnodes, neinodeids[0], visited, oanodetype + 2, 1,
                                              exp_edge_type)
                        print(oanodetype + 2, 'oai end node', oaendid)
                        node_type_to_id[oanodetype + 2] = oaendid
                        node_id_to_type[oaendid] = oanodetype + 2

            # Check Pcomm
            for pcommnodetype in [21, 22]:
                # -2 is pcomm/p1/p2 node type
                if pcommnodetype - 2 not in node_type_to_id.keys():
                    continue
                # -18 is ICA/M1/A1 node type
                antnodeid = pcommnodetype - 18
                if antnodeid not in node_type_to_id.keys():
                    print('no ica/mca/aca')
                    antnodeid = pcommnodetype - 14
                    if antnodeid not in node_type_to_id.keys():
                        antnodeid = pcommnodetype - 16
                        if antnodeid not in node_type_to_id.keys():
                            print('no a1/2, m1/2 and ica/mca/aca, skip')
                            continue
                if pcommnodetype - 20 not in node_type_to_id.keys():
                    print('no ica root, skip')
                    continue
                try:
                    sp = nx.shortest_path(graph, node_type_to_id[pcommnodetype - 2], node_type_to_id[antnodeid])
                    # shortest path should not include PCA/BA
                    if 18 in node_type_to_id.keys() and node_type_to_id[18] in sp:
                        print('has path from p1/2 to anterior, but through pca/ba, skip')
                        continue
                    # if p1/2 exist in deg2, remove
                    if pcommnodetype - 2 in deg2node:
                        del deg2node[deg2node.index(pcommnodetype - 2)]
                        del deg2edge[deg2edge.index(pcommnodetype - 4)]
                        print('pcomm has path, remove deg 2 search for node p1/2')

                except nx.NetworkXNoPath:
                    print(pcommnodetype, 'pcomm missing, and p1/2 deg2 search needed')
                    # no need to add deg2 for P1, as some p1/2 exist but not connect to pcomm
                    continue
                spica = nx.shortest_path(graph, node_type_to_id[antnodeid], node_type_to_id[pcommnodetype - 20])
                print('spica', spica, 'sp pos to ant', sp)
                assert len(set(spica) & set(sp)) > 0
                for pcommnodeid in sp:
                    if pcommnodeid in spica:
                        break
                print(pcommnodetype, 'pcomm/ica node id', pcommnodeid)
                node_type_to_id[pcommnodetype] = pcommnodeid
                node_id_to_type[pcommnodeid] = pcommnodetype

            # Check VA
            if 17 in node_type_to_id.keys() and graph.nodes[node_type_to_id[17]]['deg'] == 3:
                # check exisiting conf node type 15 16 compatibility
                for va_cf_type in [15, 16]:
                    if va_cf_type not in node_type_to_id.keys():
                        continue
                    try:
                        sp = nx.shortest_path(graph, node_type_to_id[17], node_type_to_id[va_cf_type])
                    except nx.NetworkXNoPath:
                        print('va through conf nodes are not connected, remove conf node va root')
                        del node_id_to_type[node_type_to_id[va_cf_type]]
                        del node_type_to_id[va_cf_type]

                if 15 not in node_type_to_id.keys():
                    # BA/VA and PCA/BA
                    visited = list(node_type_to_id.values())
                    vaendid = findmaxprob(graph, probnodes, node_type_to_id[17], visited, 15, 1)
                    print(15, 'VAL end node', vaendid, 'gt', targetnodes[vaendid])
                    node_type_to_id[15] = vaendid
                    node_id_to_type[vaendid] = 15

                if 16 not in node_type_to_id.keys():
                    visited = list(node_type_to_id.values())
                    vaendid = findmaxprob(graph, probnodes, node_type_to_id[17], visited, 16, 1)
                    print(16, 'VAR end node', vaendid, 'gt', targetnodes[vaendid])
                    node_type_to_id[16] = vaendid
                    node_id_to_type[vaendid] = 16

                # check LR
                if 15 in node_type_to_id.keys() and 16 in node_type_to_id.keys():
                    valnodeid = node_type_to_id[15]
                    varnodeid = node_type_to_id[16]
                    if graph.nodes[node_type_to_id[15]]['pos'][0] < graph.nodes[node_type_to_id[16]]['pos'][0]:
                        print('VALR swap')
                        node_type_to_id[15] = varnodeid
                        node_type_to_id[16] = valnodeid
                        node_id_to_type[valnodeid] = 16
                        node_id_to_type[varnodeid] = 15

            # in full graph
            # TODO
            if len(deg2edge):
                print('#Deg 2 search edge', deg2edge, 'node', deg2node)

            if len(node_id_to_type) != len(node_type_to_id):
                print('len(node_id_to_type)!=len(node_type_to_id), conflict of nodes')
                node_id_to_type = {node_type_to_id[i]: i for i in node_type_to_id}

            ##########################################
            # apply confident predictions to confoutput
            confoutput = {}
            confoutput['nodes'] = np.zeros(probnodes.shape)
            confoutput['edges'] = np.zeros(probedges.shape)
            for nodei in range(confoutput['nodes'].shape[0]):
                if nodei in node_id_to_type.keys() and node_id_to_type[nodei] not in deg2node:
                    confoutput['nodes'][nodei][node_id_to_type[nodei]] = 1
                else:
                    if nodei in node_id_to_type.keys() and node_id_to_type[nodei] in deg2node:
                        print('nodei in deg2node', node_id_to_type[nodei], 'skip setting node id')
                    if prednodes[nodei] == 0:
                        confoutput['nodes'][nodei] = probnodes[nodei]
                    else:
                        # set as non type if original max prob not nontype
                        confoutput['nodes'][nodei][0] = 1

            # fill edge according to node
            for nodetypei in range(len(nodeconnection)):
                if len(nodeconnection[nodetypei]) == 0:
                    continue
                if nodetypei not in node_id_to_type.values():
                    continue
                for branchnodetypei in nodeconnection[nodetypei]:
                    if branchnodetypei not in node_id_to_type.values():
                        continue
                    edgetype = matchvestype(nodetypei, branchnodetypei)
                    if edgetype in [7, 8] and 11 in deg2edge and branchnodetypei in deg2node:
                        print('edge', edgetype, 'needs deg 2 prediction, set edge to distal type')
                        edgetype += 2
                    if edgetype in [17, 18] and edgetype in deg2edge and branchnodetypei in deg2node:
                        print('edge', edgetype, 'needs deg 2 prediction, set edge to distal type')
                        edgetype += 2
                    try:
                        sp = nx.shortest_path(graph, node_type_to_id[nodetypei], node_type_to_id[branchnodetypei])
                    except nx.NetworkXNoPath:
                        print('no shortest path  between connection nodes, skip', nodetypei, branchnodetypei)
                    # print('sp',sp)
                    for spi in range(1, len(sp)):
                        edgei = findedgeid(graph, sp[spi - 1], sp[spi])
                        if edgei != -1:
                            confoutput['edges'][edgei][edgetype] = 1
                        else:
                            print('cannot find id for edge', sp[spi - 1], sp[spi])

            # fill additional edges based on node types
            for edgetype, nodetypes in edgemap.items():
                if nodetypes[0] in node_type_to_id.keys() and nodetypes[1] in node_type_to_id.keys():
                    try:
                        sp = nx.shortest_path(graph, node_type_to_id[nodetypes[0]], node_type_to_id[nodetypes[1]])
                        for spid in range(1, len(sp)):
                            spi = sp[spid - 1]
                            if spi in list(set(node_type_to_id.keys()) - set(
                                    [node_type_to_id[nodetypes[0]], node_type_to_id[nodetypes[1]]])):
                                # print(edgetype,nodetypes,'path through other confident nodes')
                                break
                            spj = sp[spid]
                            edgei = findedgeid(graph, spi, spj)
                            if edgei != -1:
                                print(edgetype, nodetypes, 'label edgei', edgei, 'edgetype', edgetype)
                                confoutput['edges'][edgei][edgetype] = 1
                            else:
                                print('cannot find edge id, possible?', spi, spj)
                    except nx.NetworkXNoPath:
                        print('no path between edgetype,nodetypes', edgetype, nodetypes)
                        continue

            # fill M2 A2 P2
            # add keyids to visted list to avoid propogate through acomm pcomm
            keyids = []
            for nodeids in node_type_to_id.values():
                keyids.append(nodeids)
            print('keyids', node_id_to_type, node_type_to_id)
            print('keyids', keyids)
            for fi in fillmap:
                if fi[1] not in node_type_to_id.keys() or fi[0] not in node_type_to_id.keys():
                    continue
                if nx.has_path(graph, node_type_to_id[fi[1]], node_type_to_id[fi[0]]):
                    sp = nx.shortest_path(graph, node_type_to_id[fi[1]], node_type_to_id[fi[0]])
                else:
                    continue
                assert len(sp) >= 2
                fillnodes, filledges = findallnei(graph, node_type_to_id[fi[1]], sp[:2] + keyids)
                print('fill node/edge', fi, fillnodes, filledges)
                # node set 0
                for nodeid in fillnodes:
                    if np.argmax(confoutput['nodes'][nodeid]) != 0:
                        print('node already assigned', nodeid, 'no fill needed')
                        continue
                    nodez = np.zeros((BOITYPENUM))
                    nodez[0] = 1
                    confoutput['nodes'][nodeid] = nodez
                # edge set to edgetype
                edgetype = fi[2]
                for edgeid in filledges:
                    edgei = findedgeid(graph, edgeid[0], edgeid[1])
                    if np.argmax(confoutput['edges'][edgei]) != 0:
                        print(edgei, 'assign', edgetype, 'edge already assigned to ',
                              np.argmax(confoutput['edges'][edgei]))
                        if np.argmax(confoutput['edges'][edgei]) in [9, 10] and edgetype in [9, 10]:
                            print('ERR NEED CORR, A2 touch')
                            # a2LR touches, compare dist to A1/2 L and R
                            edgenodei = list(graph.edges())[edgei][0]
                            cdistL = nodedist(graph, edgenodei, node_type_to_id[5])
                            cdistR = nodedist(graph, edgenodei, node_type_to_id[6])
                            edgez = np.zeros((VESTYPENUM))
                            if cdistL < cdistR:
                                print('A2LR touch, set to A2L')
                                edgez[9] = 1
                            else:
                                print('A2LR touch, set to A2R')
                                edgez[10] = 1
                            confoutput['edges'][edgei] = edgez
                    else:
                        edgez = np.zeros((VESTYPENUM))
                        edgez[edgetype] = 1
                        confoutput['edges'][edgei] = edgez

            # fill remaining with prob if not exist, if pred non-type, force to set zero type
            for edgei in range(confoutput['edges'].shape[0]):
                # if unset, check connection to nodetype M12 A12 P12, set to closest
                if np.max(confoutput['edges'][edgei]) == 0:
                    # if prednodes[nodei]==0:
                    cprobedge = probedges[edgei]
                    if np.argmax(cprobedge) == 12:
                        cprobedge[5] += cprobedge[12]
                        cprobedge[12] = 0
                    if np.argmax(cprobedge) == 13:
                        cprobedge[6] += cprobedge[13]
                        cprobedge[13] = 0
                    enodei = list(graph.edges())[edgei][0]
                    try:
                        sp = nx.shortest_path(graph, enodei, node_type_to_id[7])
                        if node_type_to_id[3] not in sp:
                            print(edgei, 'has loop, remaining edge set to m2l')
                            zprobedge = np.zeros((VESTYPENUM))
                            zprobedge[5] = 1
                            confoutput['edges'][edgei] = zprobedge
                            continue
                    except:
                        pass
                    try:
                        sp = nx.shortest_path(graph, enodei, node_type_to_id[8])
                        if node_type_to_id[4] not in sp:
                            print(edgei, 'has loop, remaining edge set to m2r')
                            zprobedge = np.zeros((VESTYPENUM))
                            zprobedge[6] = 1
                            confoutput['edges'][edgei] = zprobedge
                            continue
                    except:
                        pass
                    try:
                        sp = nx.shortest_path(graph, enodei, node_type_to_id[5])
                        if node_type_to_id[3] not in sp and node_type_to_id[4] not in sp:
                            print(edgei, 'has loop, remaining edge set to a2l')
                            zprobedge = np.zeros((VESTYPENUM))
                            zprobedge[7] = 1
                            confoutput['edges'][edgei] = zprobedge
                            continue
                    except:
                        pass
                    try:
                        sp = nx.shortest_path(graph, enodei, node_type_to_id[6])
                        if node_type_to_id[3] not in sp and node_type_to_id[4] not in sp:
                            print(edgei, 'has loop, remaining edge set to a2r')
                            zprobedge = np.zeros((VESTYPENUM))
                            zprobedge[8] = 1
                            confoutput['edges'][edgei] = zprobedge
                            continue
                    except:
                        pass
                    try:
                        sp = nx.shortest_path(graph, enodei, node_type_to_id[19])
                        if node_type_to_id[18] not in sp and node_type_to_id[3] not in sp:
                            print(edgei, 'has loop, remaining edge set to p2l')
                            zprobedge = np.zeros((VESTYPENUM))
                            zprobedge[19] = 1
                            confoutput['edges'][edgei] = zprobedge
                            continue
                    except:
                        pass
                    try:
                        sp = nx.shortest_path(graph, enodei, node_type_to_id[20])
                        if node_type_to_id[18] not in sp and node_type_to_id[4] not in sp:
                            print(edgei, 'has loop, remaining edge set to p2r')
                            zprobedge = np.zeros((VESTYPENUM))
                            zprobedge[20] = 1
                            confoutput['edges'][edgei] = zprobedge
                            continue
                    except:
                        pass

                    # if not connected to any of the 1/2 branch, set to edge pred value
                    confoutput['edges'][edgei] = cprobedge
                    if np.argmax(cprobedge) != 0:
                        print('remaining edge', edgei, np.argmax(cprobedge))
            # else:
            #	#set as non type if original max prob not nontype
            #	confoutput['edges'][edgei][0] = 1

            refoutputs.append(copy.deepcopy(confoutput))
            print('node type to id', node_type_to_id)

        endtime = datetime.datetime.now()
        reftime = (endtime - starttime).total_seconds()
        print('time per case', reftime)
        print('confnodes', fitacc, 'err', errfitacc)
        return refoutputs[0]