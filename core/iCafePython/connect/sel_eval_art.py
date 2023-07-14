import numpy as np
from ..swcnode import SWCNode
from ..snakelist import SnakeList
from ..snake import Snake
from ..point3d import Point3D
import networkx as nx
import copy

def trimSnake(csnake, ves_head, ves_end):
    min_dist_a = np.inf
    min_a_id = None
    min_dist_b = np.inf
    min_b_id = None

    for pti in range(csnake.NP):
        dist_a = csnake[pti].pos.dist(ves_head)
        if dist_a < min_dist_a:
            min_dist_a = dist_a
            min_a_id = pti
        dist_b = csnake[pti].pos.dist(ves_end)
        if dist_b < min_dist_b:
            min_dist_b = dist_b
            min_b_id = pti
    print('trim', min_a_id, min_b_id, csnake)
    if min_a_id>min_b_id:
        print('reverse snake')
        csnake.reverseSnake()
        min_a_id, min_b_id = min_b_id, min_a_id

    trim_csnake = csnake.copy()
    if min_b_id != csnake.NP - 1:
        trim_csnake.trimSnake(min_b_id, reverse=False)
    else:
        trim_csnake.addSWC(ves_end, trim_csnake[-1].rad)
    if min_a_id != 0:
        trim_csnake.trimSnake(min_a_id, reverse=True)
    else:
        trim_csnake.insert(0, SWCNode(ves_head, trim_csnake[0].rad))
    return trim_csnake



def selArtFromGraph(ves_tree_snakelist,ves_A,ves_B,ves_head,ves_end):
    node_G_base = ves_tree_snakelist.nodeGraph()
    # result snake
    csnake = Snake()
    match_start_type = None
    match_end_type = None

    match_snakei, match_ptidi, match_dist_i, _ = ves_tree_snakelist.matchPt(ves_head, thres_rad=12)
    if match_snakei!=-1:
        match_start_type = 'end'
    match_snakej, match_ptidj, match_dist_j, _ = ves_tree_snakelist.matchPt(ves_end, thres_rad=12)
    if match_snakej!=-1:
        match_end_type = 'end'
    if match_snakei!=-1 and match_snakei == match_snakej:
        print('Match same branch',match_snakei)
        return trimSnake(ves_tree_snakelist[match_snakei], ves_head, ves_end)

    if match_snakei == -1:
        match_snakei, match_ptidi, match_dist_i, _ = ves_tree_snakelist.matchPt(ves_B, thres_rad=8)
        if match_snakei == -1:
            thres_rad = 5
            print('ves_head/B not found, start search with thres',thres_rad)
            while thres_rad < 100:
                thres_rad += 3
                print('find head pt with thres', thres_rad)
                match_snakei, match_ptidi, match_dist_i, _ = ves_tree_snakelist.matchPt(ves_B, thres_rad=thres_rad)
                if match_snakei != -1:
                    print('match ves_B', match_snakei, match_ptidi)
                    match_start_type = 'middle'
                    break
                match_snakei, match_ptidi, match_dist_i, _ = ves_tree_snakelist.matchPt(ves_head, thres_rad=thres_rad)
                if match_snakei != -1:
                    print('match ves_head', match_snakei, match_ptidi)
                    match_start_type = 'end'
                    break
        else:
            match_start_type = 'middle'
            print('ves_head not found, ves_B found')


    if match_snakej == -1:
        match_snakej, match_ptidj, match_dist_j, _ = ves_tree_snakelist.matchPt(ves_A, thres_rad=8)
        if match_snakej == -1:
            thres_rad = 5
            print('ves_end/A not found, start search with thres',thres_rad)
            while thres_rad < 100:
                print('find end pt with thres', thres_rad)
                thres_rad += 3
                match_snakej, match_ptidj, match_dist_j, _ = ves_tree_snakelist.matchPt(ves_A, thres_rad=thres_rad)
                if match_snakej != -1:
                    print('match ves_A', match_snakej, match_ptidj)
                    match_end_type = 'middle'
                    break
                match_snakej, match_ptidj, match_dist_j, _ = ves_tree_snakelist.matchPt(ves_end, thres_rad=thres_rad)
                if match_snakej != -1:
                    print('match ves_end', match_snakej, match_ptidj)
                    match_end_type = 'end'
                    break
        else:
            match_end_type = 'middle'
            print('ves_end not found, ves_A found')

    node_start = ves_tree_snakelist[match_snakei][match_ptidi].pos
    node_end = ves_tree_snakelist[match_snakej][match_ptidj].pos
    print('match snake pt', match_snakei, match_ptidi, 'to', match_snakej, match_ptidj)
    if match_snakei == -1:
        print('no exist snakei near ves_head', ves_head,' Abort')
        return csnake
    if match_snakej == -1:
        print('no exist snakei near ves_end', ves_end,' Abort')
        return csnake

    csnake, snode_path = iter_graph_find_snake(node_G_base, ves_tree_snakelist, node_start, node_end)
    # add ves A as additional snake
    # nodei = len(node_G.graph['pt_map'])
    # node_G.add_node(nodei, pos=ves_A.lst(), rad=1)
    # hash_pos = ves_A.hashPos()
    # node_G.graph['pt_map'][hash_pos] = nodei
    # csnake = Snake()
    # csnake.addSWC(ves_A)
    # ves_tree_snakelist.addSnake(csnake)


    graph_search_end_node = 1
    if graph_search_end_node:
        print(match_start_type,match_end_type)
        if match_start_type == 'middle':
            add_snake, add_snode_path = iter_graph_find_snake(node_G_base, ves_tree_snakelist, node_start, ves_head)
            if add_snake.NP>0 and add_snake.length<1.5*node_start.dist(ves_head):
                print('add remaining start snake',add_snake,'to',csnake)
                csnake.mergeSnakeA(add_snake)
        if match_end_type=='middle':
            add_snake, add_snode_path = iter_graph_find_snake(node_G_base, ves_tree_snakelist, node_end, ves_end)
            if add_snake.NP>0 and add_snake.length<1.5*node_end.dist(ves_end):
                print('add remaining end snake',add_snake,'to',csnake)
                csnake.mergeSnakeA(add_snake)
    else:
        # add remaining snake before node start
        if snode_path[0] < snode_path[1]:
            add_head_snake = ves_tree_snakelist[match_snakei].trimSnake(match_ptidi, copy=True)
            csnake.mergeSnake(add_head_snake, reverse=True, append=False)
        else:
            add_head_snake = ves_tree_snakelist[match_snakei].trimSnake(match_ptidi, reverse=True, copy=True)
            csnake.mergeSnake(add_head_snake, reverse=False, append=False)
        # add remaining snake after node end
        if snode_path[-1] < snode_path[-2]:
            add_tail_snake = ves_tree_snakelist[match_snakej].trimSnake(match_ptidj, copy=True)
            csnake.mergeSnake(add_tail_snake, reverse=True, append=True)
        else:
            add_tail_snake = ves_tree_snakelist[match_snakej].trimSnake(match_ptidj, reverse=True, copy=True)
            csnake.mergeSnake(add_tail_snake, reverse=False, append=True)

    print('csnake', csnake)
    trim_csnake = trimSnake(csnake, ves_head, ves_end)
    return trim_csnake

def iter_graph_find_snake(node_G_base,ves_tree_snakelist,node_start,node_end):
    csnake = Snake()
    if node_end.hashPos() not in node_G_base.graph['pt_map']:
        match_snakei, match_ptidi, match_dist_i, _ = ves_tree_snakelist.matchPt(node_end, thres_rad=50)
        if match_snakei==-1:
            print('no snake near ves end')
            return csnake,[]
        print('searching node_end',node_end,'to',ves_tree_snakelist[match_snakei][match_ptidi].pos,'dist',match_dist_i)
        node_end = ves_tree_snakelist[match_snakei][match_ptidi].pos

    if node_start.hashPos() not in node_G_base.graph['pt_map']:
        match_snakej, match_ptidj, match_dist_j, _ = ves_tree_snakelist.matchPt(node_start, thres_rad=50)
        if match_snakej == -1:
            print('no snake near ves start')
            return csnake,[]
        print('searching node_start',node_end,'to',ves_tree_snakelist[match_snakej][match_ptidj].pos,'dist',match_dist_j)
        node_start = ves_tree_snakelist[match_snakej][match_ptidj]

    search_range = 10
    while search_range <= 100:
        # clear copy with no edges
        node_G = copy.deepcopy(node_G_base)

        for snakei in range(ves_tree_snakelist.NSnakes):
            for pti in [0, -1]:
                cpos = ves_tree_snakelist._snakelist[snakei][pti].pos
                nodei = node_G.graph['pt_map'][cpos.hashPos()]
                exclude_snakeids = [snakei]
                match_cands = ves_tree_snakelist.matchPts(cpos, search_range, exclude_snakeids)
                for snakej, ptj, cdist in match_cands:
                    posj = ves_tree_snakelist._snakelist[snakej][ptj].pos
                    nodej = node_G.graph['pt_map'][posj.hashPos()]
                    if not nx.has_path(node_G, nodei, nodej) or nx.shortest_path_length(node_G, nodei, nodej, weight ='dist')>posj.dist(cpos):
                        node_G.add_edge(nodei, nodej, dist=cdist, snakei=snakei, pti=pti, snakej=snakej, ptj=ptj)
        if not nx.has_path(node_G, node_G.graph['pt_map'][node_start.hashPos()],
                           node_G.graph['pt_map'][node_end.hashPos()]):
            search_range += 10
            print('no path, search range increase to', search_range)
        else:
            print('found path')
            break

    if not nx.has_path(node_G, node_G.graph['pt_map'][node_start.hashPos()],
                       node_G.graph['pt_map'][node_end.hashPos()]):
        print('no path')
    else:
        print('node_start', node_start, 'node_end', node_end)
        snode_path = nx.shortest_path(node_G, node_G.graph['pt_map'][node_start.hashPos()],
                                      node_G.graph['pt_map'][node_end.hashPos()], weight='dist')
        print('path has', len(snode_path), 'nodes')
        for si in range(len(snode_path)):
            cpos = Point3D(node_G.nodes[snode_path[si]]['pos'])
            crad = node_G.nodes[snode_path[si]]['rad']
            csnake.addSWC(cpos, crad, cid=si)

    print('snae',csnake[0],csnake[-1])
    return csnake, snode_path