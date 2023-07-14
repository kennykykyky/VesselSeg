import numpy as np
import copy
from skimage.morphology import skeletonize,skeletonize_3d
import matplotlib.pyplot as plt
from .skeleton_utils import dist, minInd,repaint, findPath
import itertools
import networkx as nx
from ..point3d import Point3D
from ..swcnode import SWCNode
from ..snake import Snake
from ..snakelist import SnakeList
from .cut import cutSnakeList

def constructSkeleton(self,simg):
    binary_img = binImg(simg, sel_main_tree = 0)
    plt.title('bin img')
    plt.imshow(np.max(binary_img, axis=2))
    plt.show()
    G = skeleGraph(binary_img)

    all_traces = findTraceFromGraph(G)

    all_traces_snakelist = constructSnakeList(all_traces, simg)
    all_traces_snakelist = all_traces_snakelist.removeShort(3)
    self.writeSWC('seg_raw', all_traces_snakelist)

    all_traces_cut_snakelist = cutSnakeList(self.tifimg, all_traces_snakelist, nstd=5)

    seg_ves_snakelist = all_traces_cut_snakelist
    #remove duplicate trace by labeled map, but might also remove branches only touching head tail few points
    #seg_ves_snakelist = all_traces_cut_snakelist.trimDuplicateSnake(self.I['s.whole'].shape)

    seg_ves_snakelist.removeDuplicatePts()

    seg_ves_snakelist = seg_ves_snakelist.removeShort(1)
    for snakei in range(seg_ves_snakelist.NSnakes):
        if seg_ves_snakelist[snakei].NP<3:
            seg_ves_snakelist[snakei].resampleSnake(3)

    self.writeSWC('seg_ves', seg_ves_snakelist)
    return seg_ves_snakelist


def constructSkeleton_V1(self, simg):
    sel_main_tree = 0
    binary_img = binImg(simg, sel_main_tree)

    plt.title('bin img')
    plt.imshow(np.max(binary_img, axis=2))
    plt.show()

    G = skeleGraph(binary_img)

    all_traces = findTraceFromGraph(G)

    # generate idMap
    all_traces_snakelist = constructSnakeList(all_traces, simg)
    self.writeSWC('seg_raw', all_traces_snakelist)

    idMap = all_traces_snakelist.idMap(self.tifimg.shape)

    all_traces_valid = mergeTraces(all_traces, idMap, self.I['s.whole'],self.tifimg.shape)

    # export
    all_traces_valid_snakelist = constructSnakeList(all_traces_valid, simg)

    # write seeds
    if 0:  # len(self.seeds)==0:
        self.setSeedsSnakeList(all_traces_valid_snakelist)
        self.writeSeeds()
        print('write %d seeds' % (len(self.seeds)))

    all_traces_cut_snakelist = cutSnakeList(self.tifimg, all_traces_valid_snakelist, 5)
    all_traces_cut_snakelist.autoMerge()
    all_traces_cut_snakelist = all_traces_cut_snakelist.removeShort(3)

    all_traces_trim_snakelist = all_traces_cut_snakelist.trimDuplicateSnake(self.I['s.whole'].shape)
    all_traces_trim_snakelist.autoMerge()
    all_traces_trim_snakelist = all_traces_trim_snakelist.removeShort(3)

    self.writeSWC('seg_ves', all_traces_trim_snakelist.resampleSnakes(3))

    #branch
    seg_ves_snakelist = all_traces_trim_snakelist
    seg_ves_snakelist.autoBranch()
    seg_ves_snakelist.removeDuplicatePts()
    seg_ves_snakelist.assignDeg()
    seg_ves_snakelist.autoTransform()
    # _ = seg_ves_snakelist.resampleSnakes(1)
    # main_snakelist = seg_ves_snakelist.mainArtTree()
    seg_ves_snakelist.assignDeg()
    self.writeSWC('seg_ves_branch', seg_ves_snakelist)

    return seg_ves_snakelist


def constructSkeletonV2(self,simg):
    binary_img = binImg(simg, sel_main_tree = 0)
    plt.title('bin img')
    plt.imshow(np.max(binary_img, axis=2))
    plt.show()
    G = skeleGraph(binary_img)

    all_traces = findTraceFromGraph(G)

    all_traces_snakelist = constructSnakeList(all_traces, simg)
    self.writeSWC('seg_raw', all_traces_snakelist.removeShort(3))

    all_traces_cut_snakelist = cutSnakeList(self.tifimg, all_traces_snakelist, nstd=5)
    all_traces_cut_snakelist.autoMerge()
    all_traces_cut_snakelist = all_traces_cut_snakelist.removeShort(3)

    all_traces_trim_snakelist = all_traces_cut_snakelist.trimDuplicateSnake(self.I['s.whole'].shape)
    all_traces_trim_snakelist.autoMerge()
    all_traces_trim_snakelist = all_traces_trim_snakelist.removeShort(3)

    # branch
    all_traces_trim_snakelist.removeDuplicatePts()
    all_traces_trim_snakelist.assignDeg()
    all_traces_trim_snakelist.autoTransform()
    seg_ves_snakelist = all_traces_trim_snakelist.resampleSnakes(3)
    seg_ves_snakelist.assignDeg()
    self.writeSWC('seg_ves', seg_ves_snakelist)
    return seg_ves_snakelist


def lowThresAddSkeleton(self,seg_ves_snakelist,thres=0.1):
    simg_low = self.I['s.whole'] > thres

    sel_main_tree = 0
    binary_img_low = binImg(simg_low, sel_main_tree)

    plt.title('bin img')
    plt.imshow(np.max(binary_img_low, axis=2))
    plt.show()

    G_low = skeleGraph(binary_img_low)

    all_traces_low = findTraceFromGraph(G_low)

    # generate idMap
    all_traces_snakelist_low = constructSnakeList(all_traces_low, simg_low)

    combine_traces_snakelist = SnakeList()
    combine_traces_snakelist.addSnakeList(seg_ves_snakelist)
    combine_traces_snakelist.addSnakeList(all_traces_snakelist_low)

    snake_npts = [seg_ves_snakelist[i].NP for i in range(seg_ves_snakelist.NSnakes)]
    snake_order = np.argsort(snake_npts)[::-1]
    snake_order = snake_order.tolist() + np.arange(seg_ves_snakelist.NSnakes, combine_traces_snakelist.NSnakes).tolist()
    combine_traces_tdup_snakelist = combine_traces_snakelist.trimDuplicateSnake(self.I['s.whole'].shape, snake_order)

    all_traces_combine = combine_traces_tdup_snakelist.toTraceList()

    idMap = combine_traces_tdup_snakelist.idMap(self.tifimg.shape)

    all_traces_merged = mergeTraces(all_traces_combine, idMap, self.I['s.whole'], self.tifimg.shape)
    all_traces_valid_snakelist = constructSnakeList(all_traces_merged, simg_low)

    all_traces_cut_snakelist = cutSnakeList(self.tifimg, all_traces_valid_snakelist, 5)
    all_traces_cut_snakelist.autoMerge()
    all_traces_cut_snakelist = all_traces_cut_snakelist.removeShort(3)

    all_traces_trim_snakelist = all_traces_cut_snakelist.trimDuplicateSnake(self.I['s.whole'].shape)
    all_traces_trim_snakelist.autoMerge()
    all_traces_trim_snakelist = all_traces_trim_snakelist.removeShort(3)

    self.writeSWC('seg_ves2', all_traces_trim_snakelist.resampleSnakes(3))
    return all_traces_trim_snakelist

#break loop in the graph to form tree structure
def segTree(main_snakelist,thres_dist=20):
    def addBranchEdge(snakelist, branch_G, thres_dist=20):
        for snakei in range(snakelist.NSnakes):
            # check head and tails
            for ckpt in [0, -1]:
                cpos = snakelist[snakei][ckpt].pos
                match_snakeid, match_ptid, match_dist, match_rad = snakelist.matchPt(cpos, snakei, thres_dist)
                if match_snakeid != -1:
                    if match_dist < thres_dist:
                        print(snakei, 'head' if ckpt == 0 else 'tail', 'match to', match_snakeid, 'dist', match_dist)
                        closs = match_dist
                        branch_G.add_edge(snakei, match_snakeid, dist=match_dist, loss=closs, snakei=snakei, pti=ckpt,
                                          snakej=match_snakeid, ptj=match_ptid)
        return branch_G

    branch_G = main_snakelist.branchGraph()
    branch_G = addBranchEdge(main_snakelist, branch_G, thres_dist)

    tree_G = nx.minimum_spanning_tree(branch_G, 'loss')
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(tree_G, font_size=13, node_size=[20 + branch_G.nodes[i]['NP'] for i in branch_G.nodes()],
                     pos={i: tree_G.nodes[i]['pos'][:2] for i in tree_G.nodes()},
                     node_color='r')
    plt.show()

    print('Nodes before tree',len(tree_G.edges()),'after tree', len(branch_G.edges()))

    def constructSnakeFromTree(snakelist, tree_G):
        snakelist = copy.deepcopy(snakelist)
        for edgei in tree_G.edges():
            edge_item = tree_G.edges[edgei]
            snakei, pti, snakej, ptj = edge_item['snakei'], edge_item['pti'], edge_item['snakej'], edge_item['ptj']
            if edge_item['dist'] != 0:
                print('branch', snakei, pti, snakej, ptj, edge_item['dist'])
                snakelist[snakei].branchSnake(snakelist[snakej][ptj], pti)
        # snakelist.autoMerge()
        # snakelist = snakelist.mainArtTree(10,5)
        return snakelist

    main_tree_snakelist = constructSnakeFromTree(main_snakelist, tree_G)
    #main_tree_snakelist = main_tree_snakelist.mainArtTree(10, 15)
    main_tree_snakelist.autoMerge()
    main_tree_snakelist.assignDeg()
    main_tree_snakelist.autoTransform()
    #main_tree_snakelist.removeDuplicatePts()
    main_tree_snakelist.removeSelfLoop()
    main_tree_snakelist.assignDeg()
    return main_tree_snakelist


def binImg(simg,sel_main_tree):
#skeleton = skeletonize(simg,method='lee')
    skeleton = skeletonize_3d(simg)
    #import skimage
    #connect_region,label_num = skimage.measure.label(skeleton,connectivity=3,return_num=True)
    from scipy import ndimage
    label_im, nb_labels = ndimage.label(skeleton,np.ones((3,3,3)))
    sizes = ndimage.sum(skeleton, label_im, range(nb_labels + 1))

    if sel_main_tree:
        maxid = np.argmax(sizes)
        binary_img = label_im==maxid
    else:
        binary_img = label_im
    return binary_img

def skeleGraph(binary_img):
    cid = 0
    G = nx.Graph()
    pos_dict = {}

    pos = np.argwhere(binary_img)
    pos = pos.tolist()
    pos.sort(key=lambda x: (x[2],x[1],x[0]))
    pos = np.array(pos)

    pos_z = [[] for z in range(binary_img.shape[2])]
    for i in pos:
        pos_z[i[2]].append(i)

    for posid in range(len(pos)):
        if posid%200==0:
            print('\rConstructing skeleton graph with nodes',posid,'/',len(pos),end='')
        posi = pos[posid]
        G.add_node(cid,pos=posi)
        nei_pos = list(itertools.chain.from_iterable([pos_z[posi[2]+ofi] \
                       for ofi in range(-1,2) if posi[2]+ofi>=0 and posi[2]+ofi<len(pos_z)]))
        dev_pos_key = None
        for nei_posi in nei_pos:
            nei_posi_key = '-'.join((nei_posi).astype(str))

            cdist = dist(nei_posi,posi)
            if nei_posi_key in pos_dict and cdist<2:
                dev_pos_key = '-'.join((nei_posi).astype(str))
                # print('add',cid,pos_dict[dev_pos_key])
                G.add_edge(pos_dict[dev_pos_key], cid, dist=cdist)

        pos_dict['-'.join(posi.astype(str))] = cid
        cid += 1
    return G

def findTraceFromGraph(G):
    # trace from deg 1
    # found ids
    fd_ids = set()

    # find first deg1 node
    def getDeg1(G, fd_ids):
        for s in G.degree():
            if s[1] == 1 and s[0] not in fd_ids:
                cid = s[0]
                # print('start from deg1 node',cid)
                return cid
        return -1

    cid = getDeg1(G, fd_ids)
    if cid != -1:
        heads = [cid]
    all_traces = []

    while len(heads):
        cid = heads.pop(0)
        # print('start tracing',cid)
        fd_ids.add(cid)
        trace = [G.nodes[cid]['pos']]
        while 1:
            nei_ids = list(set([n for n in G.neighbors(cid)]) - set(fd_ids))
            # print(cid,nei_ids)
            if len(nei_ids) == 0:
                # print('end tracing',len(trace))
                if len(trace) > 1:
                    all_traces.append(copy.deepcopy(trace))
                break
            next_id = nei_ids[0]
            trace.append(G.nodes[next_id]['pos'])
            if len(nei_ids) > 1:
                heads.extend(nei_ids[1:])
                # print('new head',heads)
            fd_ids.add(cid)
            cid = next_id
        if len(heads) == 0:
            cid = getDeg1(G, fd_ids)
            # print('no head, find deg1',cid)
            if cid != -1:
                heads = [cid]

    return all_traces


def mergeTraces(all_traces,idMap,simg,box):
    avail_traces = np.ones((len(all_traces)))
    replaced_ids = np.arange(len(all_traces))
    DEBUG = 1
    for tracei in range(len(all_traces)):
        if avail_traces[tracei] == 0:
            continue
        # search head
        pos_list = [0, -1]
        while len(pos_list) > 0:
            cid = pos_list.pop()
            cpos = all_traces[tracei][cid]
            mindist, (min_traceid, min_ptid) = minInd(cpos, idMap, tracei)
            if DEBUG:
                print(tracei, cpos, mindist, (min_traceid, min_ptid))
            if min_traceid is None:
                continue
            if cid == 0:
                if min_ptid == 0:
                    # merge head with head
                    all_traces[tracei] = all_traces[min_traceid][::-1] + all_traces[tracei]
                    replaced_ids[min_traceid] = tracei
                    repaint(idMap, all_traces, tracei)
                    avail_traces[min_traceid] = 0
                    pos_list.append(0)
                    if DEBUG:
                        print('head of', tracei, 'merge to head of', min_traceid, 'new head', all_traces[tracei][0])
                elif min_ptid == len(all_traces[min_traceid]) - 1:
                    # merge head with tail
                    all_traces[tracei] = all_traces[min_traceid] + all_traces[tracei]
                    replaced_ids[min_traceid] = tracei
                    repaint(idMap, all_traces, tracei)
                    avail_traces[min_traceid] = 0
                    pos_list.append(0)
                    if DEBUG:
                        print('head of', tracei, 'merge to tail of', min_traceid, 'new head', all_traces[tracei][0])
                else:
                    # if mindist<5
                    # create branch
                    print('le',len(all_traces),tracei,min_traceid,min_ptid)
                    path = findPath(simg,box,all_traces[min_traceid][min_ptid], all_traces[tracei][0])
                    if len(path) > 0:
                        if DEBUG:
                            print('create head branch', path)
                        tail_direct = Point3D(path[0] - all_traces[tracei][0])
                        follow_direct = Point3D(all_traces[tracei][0] - all_traces[tracei][-1])
                        if tail_direct.prod(follow_direct) > 0:
                            all_traces[tracei] = path + all_traces[tracei]
                        else:
                            if DEBUG:
                                print(path, all_traces[min_traceid][min_ptid], all_traces[tracei][0], tail_direct,
                                  follow_direct, tail_direct.prod(follow_direct), 'skip connect')
                    else:
                        pass
            elif cid == -1:
                if min_ptid == 0:
                    # merge tail with head
                    all_traces[tracei] = all_traces[tracei] + all_traces[min_traceid]
                    replaced_ids[min_traceid] = tracei
                    repaint(idMap, all_traces, tracei)
                    avail_traces[min_traceid] = 0
                    pos_list.append(-1)
                    if DEBUG:
                        print('tail of', tracei, 'merge to head of', min_traceid, 'new tail', all_traces[tracei][-1])
                elif min_ptid == len(all_traces[min_traceid]) - 1:
                    # merge tail with tail
                    all_traces[tracei] = all_traces[tracei] + all_traces[min_traceid][::-1]
                    replaced_ids[min_traceid] = tracei
                    repaint(idMap, all_traces, tracei)
                    avail_traces[min_traceid] = 0
                    pos_list.append(-1)
                    if DEBUG:
                        print('tail of', tracei, 'merge to tail of', min_traceid, 'new head', all_traces[tracei][-1])

                else:
                    # if mindist<5
                    # create branch
                    print('le',len(all_traces),tracei,min_traceid,min_ptid)
                    path = findPath(simg, box, all_traces[tracei][-1], all_traces[min_traceid][min_ptid])
                    if len(path) > 0:
                        if DEBUG:
                            print('create tail branch', path)
                        tail_direct = Point3D(all_traces[tracei][-1] - path[-1])
                        follow_direct = Point3D(all_traces[tracei][0] - all_traces[tracei][-1])
                        if tail_direct.prod(follow_direct) > 0:
                            all_traces[tracei] = all_traces[tracei] + path
                        else:
                            if DEBUG:
                                print(path, all_traces[tracei][-1], all_traces[min_traceid][min_ptid],
                                      tail_direct, follow_direct, tail_direct.prod(follow_direct), 'skip connect')
                    else:
                        pass

    # generate valid list
    all_traces_valid = []
    idmap = {}
    for tracei in range(len(all_traces)):
        if avail_traces[tracei] == 1:
            idmap[len(all_traces_valid)] = tracei
            all_traces_valid.append(all_traces[tracei])
    #len(all_traces_valid)
    return all_traces_valid


def constructSnakeList(all_traces,simg):
    cid = 1
    snakelist = []
    for trace in all_traces:
        swcnodelist = []
        for pti in trace:
            pti = pti.astype(np.float64)
            cpos = Point3D(pti)
            crad = cpos.findRad(simg)
            # cpos = Point3D(pti[2],pti[1],pti[0])
            swcnodelist.append(SWCNode(cpos, crad, cid))
            cid += 1
        snakelist.append(Snake(swcnodelist))
    return SnakeList(snakelist)



