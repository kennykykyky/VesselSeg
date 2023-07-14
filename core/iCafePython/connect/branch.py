import networkx as nx
import numpy as np
from collections import Counter

# search each branch end with neigh branch ends
def findNeiBranchEnds(seg_raw_snakelist,ref_snakelist,ves_G):
    match_gt_all = []

    for csnake_id in range(seg_raw_snakelist.NSnakes):
        pti = findRepPt(ref_snakelist,seg_raw_snakelist[csnake_id],-1)
        if pti==-1:
            continue
        #print('start',seg_raw_snakelist[csnake_id][-1])
        #search neighbor branch ends near the end point of csnakeid segment, while keep neighbor segments away from start point of csnakeid segment
        nei_snake_ids = findNeiSegs(seg_raw_snakelist,csnake_id,search_range=15)
        #print(csnake_id,'nei',nei_snake_ids)
        for tsnake_id, reverse_snake in nei_snake_ids:
            if reverse_snake == True:
                ptj = findRepPt(ref_snakelist,seg_raw_snakelist[tsnake_id],-1)
            else:
                ptj = findRepPt(ref_snakelist,seg_raw_snakelist[tsnake_id],0)
            same_branch = sameVesInGraph(ves_G,pti,ptj)
            #print(tsnake_id,pti,ptj,same_branch)

            match_gt_all.append((csnake_id,tsnake_id,reverse_snake,same_branch))
    return match_gt_all

# Find neighboring branch heads/tails from a pos
def findNeiSegs(seg_raw_snakelist, csnake_id, search_range=15):
    # cpos: branch end for searching neighbors
    cpos = seg_raw_snakelist[csnake_id][-1].pos
    # hpos: the other end of the searching branch, to avoid the other end pos closer to neighbor branches
    hpos = seg_raw_snakelist[csnake_id][0].pos
    # snake set
    snake_set = set()
    for pti in range(seg_raw_snakelist[csnake_id].NP):
        snake_set.add(seg_raw_snakelist[csnake_id][pti].pos.hashPos())
    tsnake_ids = []
    for snakeid in range(seg_raw_snakelist.NSnakes):
        if seg_raw_snakelist[snakeid][0].pos.hashPos() in snake_set or \
                seg_raw_snakelist[snakeid][-1].pos.hashPos() in snake_set:
            continue
        dist1 = seg_raw_snakelist[snakeid][0].pos.dist(cpos)
        dist2 = seg_raw_snakelist[snakeid][-1].pos.dist(cpos)
        cdist = min(dist1, dist2)
        hdist1 = seg_raw_snakelist[snakeid][0].pos.dist(hpos)
        hdist2 = seg_raw_snakelist[snakeid][-1].pos.dist(hpos)
        hcdist = min(hdist1, hdist2)
        if hcdist < cdist:
            # head point is closer to targets than the cpos
            continue
        if cdist > 0 and cdist < search_range:
            if dist1 == cdist:
                # head of neighbor branch is closer
                tsnake_ids.append((snakeid, 0))
            else:
                # tail closer
                tsnake_ids.append((snakeid, 1))
    return tsnake_ids


# find snake id and the representing ptid for ending points of snake branches
def findRepPt(ref_snakelist, snake, ptid, thres=3):
    pos = snake[ptid].pos
    n_check_pts = min(5, snake.NP)
    if ptid == -1:
        pos_list = [snake[-1 - ofi].pos for ofi in range(n_check_pts)]
    elif ptid == 0:
        pos_list = [snake[ofi].pos for ofi in range(n_check_pts)]
    else:
        raise ValueError('pos')

    # find best matching snakeid from raw ves gt traces
    # matchid for each point on poslist
    pos_list_matchid = []
    # map between first appear snake and pt, used to find the representing ptid of snake to search min dist from graph
    snake_pt_map = {}
    for posi in pos_list:
        snakeid, ptid = matchSnake(ref_snakelist, posi)
        pos_list_matchid.append(snakeid)
        if snakeid not in snake_pt_map:
            snake_pt_map[snakeid] = ptid

    target_snakeid = Counter(pos_list_matchid).most_common(1)[0][0]
    # print(pos_list_matchid,snake_pt_map,target_snakeid)
    return snake_pt_map[target_snakeid]


# decide whether pti and ptj are in the same vessel from shortest path in graph
def sameVesInGraph(ves_G, pti, ptj):
    if pti == -1 or ptj == -1:
        return False
    if nx.has_path(ves_G, pti, ptj):
        dist_length = nx.shortest_path_length(ves_G, pti, ptj, weight='dist')
    else:
        return False
    start_pos = ves_G.nodes[pti]['pos']
    end_pos = ves_G.nodes[ptj]['pos']
    direct_length = start_pos.dist(end_pos)
    #print('dist_length', dist_length, 'direct_length', direct_length)
    if dist_length < direct_length * 1.2 + 3:
        return True
    else:
        return False


# find the snakeid/ptid with minimum distance to posi
def matchSnake(snakelist, posi, thres=3):
    # thres max allowed distance to gt pt
    min_dist = np.inf
    min_snakeid = None
    min_ptid = None
    for snakeid in range(snakelist.NSnakes):
        for ptid in range(snakelist[snakeid].NP):
            cdist = snakelist[snakeid][ptid].pos.dist(posi)
            if cdist < min_dist:
                min_dist = cdist
                min_snakeid = snakeid
                min_ptid = snakelist[snakeid][ptid].id
    if min_snakeid is not None and min_dist < thres:
        return min_snakeid, min_ptid
    else:
        return -1, -1
