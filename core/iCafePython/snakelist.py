from .point3d import Point3D
from .swcnode import SWCNode
from .snake import Snake
import numpy as np
import networkx as nx
from .utils.eval_utils import MOTMetric
from .definition import VESTYPENUM
import copy
from .connect.con_utils import initSnake, snakeLoss
import matplotlib.pyplot as plt
import scipy
from .utils.img_utils import paint_dist_unique
from collections import Counter
from .utils.snake_utils import paintBall
import scipy.stats
import operator

class SnakeList():
    def __init__(self,snakelist=None):
        if snakelist is None:
            self._snakelist = []
        else:
            self._snakelist = snakelist
        self.comp_map = None

    @classmethod
    def fromLists(cls,all_traces):
        cid = 1
        snakelist = []
        for trace in all_traces:
            swcnodelist = []
            for pti in trace:
                crad = 1
                pti = pti.astype(np.float64)
                #cpos = Point3D(pti[2], pti[1], pti[0])
                cpos = Point3D(pti)
                swcnodelist.append(SWCNode(cpos, crad, cid))
                cid += 1
            snakelist.append(Snake(swcnodelist))
        return cls(snakelist)

    def __repr__(self):
        return 'Snakelist with %d snakes' % (len(self._snakelist))

    def __len__(self):
        return len(self._snakelist)

    def __getitem__(self, key):
        return self._snakelist[key]

    def copy(self):
        return copy.deepcopy(self)

    @property
    def NSnakes(self):
        return len(self._snakelist)

    @property
    def NPts(self):
        tot_pt = 0
        for snakei in range(self.NSnakes):
            tot_pt += self._snakelist[snakei].NP
        return tot_pt

    @property
    def length(self):
        length_sum = 0
        for snakei in self._snakelist:
            length_sum += snakei.length
        return length_sum

    @property
    def branchProximal(self):
        proximal_group = [1,2,3,4,7,8,14,15,16,17,18]
        return self.branchByGroup(proximal_group)

    @property
    def lengthProximal(self):
        proximal_group = [1,2,3,4,7,8,14,15,16,17,18]
        return self.lengthByGroup(proximal_group)

    @property
    def volumeProximal(self):
        proximal_group = [1,2,3,4,7,8,14,15,16,17,18]
        return self.volumeByGroup(proximal_group)
    
    @property
    def totProximal(self):
        proximal_group = [1,2,3,4,7,8,14,15,16,17,18]
        return self.totByGroup(proximal_group)    

    @property
    def branchDistal(self):
        distal_group = [5, 6, 9, 10, 12, 13, 19, 20]
        return self.branchByGroup(distal_group)

    @property
    def lengthDistal(self):
        distal_group = [5, 6, 9, 10, 12, 13, 19, 20]
        return self.lengthByGroup(distal_group)
    
    @property
    def volumeDistal(self):
        distal_group = [5, 6, 9, 10, 12, 13, 19, 20]
        return self.volumeByGroup(distal_group)

    @property
    def totDistal(self):
        distal_group = [5, 6, 9, 10, 12, 13, 19, 20]
        return self.totByGroup(distal_group)  

    @property
    def branchDistalACA(self):
        distal_group = [9, 10]
        return self.branchByGroup(distal_group)

    @property
    def lengthDistalACA(self):
        distal_group = [9, 10]
        return self.lengthByGroup(distal_group)
    
    @property
    def volumeDistalACA(self):
        distal_group = [9, 10]
        return self.volumeByGroup(distal_group)

    @property
    def totDistalACA(self):
        distal_group = [9, 10]
        return self.totByGroup(distal_group)

    @property
    def branchDistalMCA(self):
        distal_group = [5, 6, 12, 13]
        return self.branchByGroup(distal_group)

    @property
    def lengthDistalMCA(self):
        distal_group = [5, 6, 12, 13]
        return self.lengthByGroup(distal_group)
    
    @property
    def volumeDistalMCA(self):
        distal_group = [5, 6, 12, 13]
        return self.volumeByGroup(distal_group)

    @property
    def totDistalMCA(self):
        distal_group = [5, 6, 12, 13]
        return self.totByGroup(distal_group)

    @property
    def branchDistalPCA(self):
        distal_group = [19, 20]
        return self.branchByGroup(distal_group)

    @property
    def lengthDistalPCA(self):
        distal_group = [19, 20]
        return self.lengthByGroup(distal_group)
    
    @property
    def volumeDistalPCA(self):
        distal_group = [19, 20]
        return self.volumeByGroup(distal_group)

    @property
    def totDistalPCA(self):
        distal_group = [19, 20]
        return self.totByGroup(distal_group)

    @property
    def branchLeft(self):
        left_group = [1, 3, 5, 7, 9, 12, 14, 17, 19, 21, 23]
        return self.branchByGroup(left_group)

    @property
    def lengthLeft(self):
        left_group = [1, 3, 5, 7, 9, 12, 14, 17, 19, 21, 23]
        return self.lengthByGroup(left_group)
    
    @property
    def volumeLeft(self):
        left_group = [1, 3, 5, 7, 9, 12, 14, 17, 19, 21, 23]
        return self.volumeByGroup(left_group)

    @property
    def totLeft(self):
        left_group = [1, 3, 5, 7, 9, 12, 14, 17, 19, 21, 23]
        return self.totByGroup(left_group)

    @property
    def branchRight(self):
        right_group = [2, 4, 6, 8, 10, 13, 15, 18, 20, 22, 24]
        return self.branchByGroup(right_group)
    
    @property
    def lengthRight(self):
        right_group = [2, 4, 6, 8, 10, 13, 15, 18, 20, 22, 24]
        return self.lengthByGroup(right_group)
    
    @property
    def volumeRight(self):
        right_group = [2, 4, 6, 8, 10, 13, 15, 18, 20, 22, 24]
        return self.volumeByGroup(right_group)

    @property
    def totRight(self):
        right_group = [2, 4, 6, 8, 10, 13, 15, 18, 20, 22, 24]
        return self.totByGroup(right_group)

    def branchByGroup(self,cgroup=None):
        if cgroup is None:
            return self.NSnakes
        branch_sum = 0
        for snakei in self._snakelist:
            if snakei.type in cgroup:
                branch_sum += 1
        return branch_sum
    
    def lengthByGroup(self,cgroup=None):
        if cgroup is None:
            return self.length()
        length_sum = 0
        for snakei in self._snakelist:
            if snakei.type in cgroup:
                length_sum += snakei.length
        return length_sum
    
    def volumeByGroup(self,cgroup=None):
        if cgroup is None:
            return self.volume()
        volume_sum = 0
        for snakei in self._snakelist:
            if snakei.type in cgroup:
                volume_sum += snakei.volume
        return volume_sum
    
    def totByGroup(self, cgroup=None):
        if cgroup is None:
            return self.tot()
        tot_sum = 0
        count = 0
        for snakei in self._snakelist:
            if snakei.type in cgroup:
                tot_sum += snakei.tot
                count += 1
        if count == 0:
            count = 1
        tot_avg = tot_sum / count
        return tot_avg

    @property
    def volume(self):
        volume_sum = 0
        for snakei in self._snakelist:
            volume_sum += snakei.volume
        return volume_sum

    @property
    def tot(self):
        tot_sum = 0
        if self.length==0:
            #print('self loop')
            return 1 
        for snakei in self._snakelist:
            tot_sum += snakei.tot * snakei.length
        return tot_sum/self.length

    @property
    def link_pts(self):
        _linked_pts = 0
        for i in range(self.NSnakes):
            _linked_pts += self._snakelist[i].link_pts
        return _linked_pts

    @property
    def mean_link_dist(self):
        dists = []
        for i in range(self.NSnakes):
            for d in self._snakelist[i].link_dist:
                if not np.isnan(d):
                    dists.append(d)
        return np.mean(dists)

    def addSnake(self,new_snake):
        self._snakelist.append(new_snake)

    def addSnakeList(self,new_snake_list):
        for snakei in range(new_snake_list.NSnakes):
            self.addSnake(new_snake_list[snakei])

    def removeSnake(self,snakeid):
        del self._snakelist[snakeid]

    def clear(self):
        self._snakelist.clear()

    def append(self,snake):
        self._snakelist.append(snake)

    def extend(self,snake):
        self._snakelist.extend(snake)

    def validSnake(self,valid_list):
        assert len(valid_list) == self.NSnakes
        valid_snakelist = SnakeList()
        for snakei in range(len(valid_list)):
            if valid_list[snakei] == 1:
                valid_snakelist.addSnake(self._snakelist[snakei])
        return valid_snakelist

    def resampleSnakes(self,gap):
        for snakei in range(self.NSnakes):
            self._snakelist[snakei] = self._snakelist[snakei].resampleSnake(gap)
        return self

    def resampleSnakesZunit(self):
        for snakei in range(self.NSnakes):
            self._snakelist[snakei] = self._snakelist[snakei].resampleSnakeZunit()

    # reverse snakes which have smaller radius at the head
    def arrangeSnakesDirection(self):
        for snakei in range(self.NSnakes):
            self._snakelist[snakei].arrangeSnakeDirection()

    def trimRange(self,box):
        valid_snakelist = SnakeList()
        for snakei in range(self.NSnakes):
            valid_snakelist.addSnake(self._snakelist[snakei].trimRange(box))
        return valid_snakelist

    def removeShort(self,thres=3,return_del_id=False):
        valid_snakelist = SnakeList()
        if return_del_id:
            del_list = []
        for snakei in range(self.NSnakes):
            if len(self._snakelist[snakei])<=thres:
                if return_del_id:
                    del_list.append(1)
                continue
            valid_snakelist.addSnake(self._snakelist[snakei])
            if return_del_id:
                del_list.append(0)
        if return_del_id:
            return valid_snakelist, del_list
        else:
            return valid_snakelist

    def removeLarge(self,thres=5):
        valid_snakelist = SnakeList()
        for snakei in range(self.NSnakes):
            if self._snakelist[snakei].arad > thres:
                continue
            valid_snakelist.addSnake(self._snakelist[snakei])
        return valid_snakelist

    #from snakelist to swc list
    def toSWCList(self):
        swclist = []
        for snakei in range(self.NSnakes):
            for pti in range(self._snakelist[snakei].NP):
                swclist.append(self._snakelist[snakei][pti])
        return swclist

    #from ves snakelist to ves list
    def toVesList(self):
        veslist = [[] for i in range(VESTYPENUM)]  # list, first of vessel type, then each snake in that type
        for snakei in range(self.NSnakes):
            ctype = self._snakelist[snakei].type
            if ctype==0 or ctype>=VESTYPENUM:
                print('unseen ctype',snakei)
            veslist[ctype].append(self._snakelist[snakei])
        return veslist

    def inSnakelist(self,pos,thres=3):
        for snakei in range(self.NSnakes):
            if self._snakelist[snakei].inSnake(pos,thres)==True:
                return True
        return False

    # refresh radius based on binary map
    def fitRad(self, simg):
        for snakeid in range(self.NSnakes):
            self._snakelist[snakeid].fitRad(simg)

    #merge snake head/tail if they are at same int position
    def autoMerge(self):
        avail_traces = np.ones((self.NSnakes))
        end_pos = {}
        for snakei in range(self.NSnakes):
            if self._snakelist[snakei].NP<2:
                continue
            for pti in [0,-1]:
                c_end_pos = self._snakelist[snakei][pti].pos.intHashPos()
                #print('checkexit',snakei,pti,c_end_pos,end_pos)
                if c_end_pos in end_pos:
                    snakej,ptj = end_pos[c_end_pos]
                    #merge snakej to snakei, invalidate snakej, update the other end of snakej
                    if pti == 0:
                        # if current point is head, insert snakej to head
                        append_snake = False
                    else:
                        append_snake = True

                    if ptj == 0:
                        reverse_merge_snake = False
                        #update tail hashmap
                        tail_pos_j = self._snakelist[snakej][-1].pos.intHashPos()
                        if append_snake:
                            end_pos[tail_pos_j] = (snakei,-1)
                        else:
                            end_pos[tail_pos_j] = (snakei,0)
                    else:
                        reverse_merge_snake = True
                        #update head hashmap
                        head_pos_j = self._snakelist[snakej][0].pos.intHashPos()
                        if append_snake:
                            end_pos[head_pos_j] = (snakei,-1)
                        else:
                            end_pos[head_pos_j] = (snakei,0)
                    #print('merge snakei',snakei,'pt',pti,'snakej',snakej,'ptj',ptj,'pos',self._snakelist[snakej][0].pos,\
                    #      'reverse_merge_snake',reverse_merge_snake,'append_snake',append_snake)
                    self._snakelist[snakei].mergeSnake(self._snakelist[snakej],reverse_merge_snake,append_snake)
                    avail_traces[snakej] = 0
                    del end_pos[c_end_pos]
                else:
                    end_pos[c_end_pos] = (snakei,pti)
        valid_snake = []
        for snakei in range(len(avail_traces)):
            if avail_traces[snakei]==1:
                valid_snake.append(self._snakelist[snakei])
        self._snakelist = valid_snake

    def autoBranch(self):
        for snakei in range(self.NSnakes):
            # check head
            cpos = self._snakelist[snakei][0].pos
            crad = self._snakelist[snakei][0].rad
            match_snakeid, match_ptid, match_dist, match_rad = self.matchPt(cpos, snakei)
            if match_dist!=0 and match_snakeid != -1 and match_ptid!=0 and match_ptid!=self._snakelist[match_snakeid].NP-1:
                if match_rad+crad > match_dist:
                    #print(snakei, 'head branch to', match_snakeid, match_ptid, 'dist', match_rad, crad, match_dist)
                    self._snakelist[snakei].branchSnake(self._snakelist[match_snakeid][match_ptid],0)
            # match end
            cpos = self._snakelist[snakei][-1].pos
            crad = self._snakelist[snakei][-1].rad
            match_snakeid, match_ptid, match_dist, match_rad = self.matchPt(cpos, snakei)
            if match_dist!=0 and match_snakeid != -1 and match_ptid!=0 and match_ptid!=self._snakelist[match_snakeid].NP-1:
                if match_rad+crad > match_dist:
                    #print(snakei, 'end branch to', match_snakeid, match_ptid, 'dist', match_rad, crad, match_dist)
                    self._snakelist[snakei].branchSnake(self._snakelist[match_snakeid][match_ptid], -1)
                    #print(self._snakelist[snakei][-1],self._snakelist[match_snakeid][match_ptid])

    #within the threshold search snakej with min loss either to merge or make branching
    def autoConnect(self,icafem,search_range=30,DEBUG=0):
        snake_avail = [1] * self.NSnakes
        for snakei in range(self.NSnakes):
            if snake_avail[snakei] == 0:
                continue
            pending_pti = [0, -1]
            while len(pending_pti):
                pti = pending_pti.pop(0)
                cpos = self._snakelist[snakei][pti].pos
                if DEBUG:
                    print('Searching',snakei,pti,'from',cpos,'+'*10)
                exclude_snakeids = [i for i in range(len(snake_avail)) if snake_avail[i]==0]
                match_cands = self.matchPts(cpos, search_range, exclude_snakeids)
                # print(match_cands)
                match_losses = []
                for snakej, ptj, cdist in match_cands:
                    if snake_avail[snakej] == 0:
                        match_losses.append(np.inf)
                        continue
                    if snakej == snakei:
                        # exclude itself
                        match_losses.append(np.inf)
                        continue

                    # also ensure pti is the closest point to make the branch
                    mindist_snakei = np.inf
                    minpti = None
                    for ptii in range(self._snakelist[snakei].NP):
                        cdistii = self._snakelist[snakei][ptii].pos.dist(self._snakelist[snakej][ptj].pos)
                        if cdistii < mindist_snakei:
                            minpti = ptii
                            mindist_snakei = cdistii
                    ptir = pti if pti == 0 else self._snakelist[snakei].NP - 1
                    if minpti != ptir:
                        match_losses.append(np.inf)
                        if DEBUG:
                            print('pti not the closest', snakei, ptir, minpti, 'remove cand',snakej, ptj, self._snakelist[snakej][ptj].pos)
                        continue

                    merge_snake_init = Snake()
                    merge_snake_init.add(self._snakelist[snakei][pti])
                    merge_snake_init.add(self._snakelist[snakej][ptj])
                    merge_snake_init = merge_snake_init.resampleSnake(1)
                    # merge_snake_ref = icafem.simpleRefSnake(merge_snake_init)
                    merge_snake_ref = merge_snake_init
                    closs = snakeLoss(icafem, merge_snake_ref) - 0.5 / search_range * (search_range - cdist)
                    match_losses.append(closs)

                for i in range(len(match_cands)):
                    if DEBUG:
                        print(list(match_cands)[i], match_losses[i])
                if len(match_losses) <= 1 or np.min(match_losses) > -0.3:
                    continue
                cand_best = np.argmin(match_losses)
                snakej = list(match_cands)[cand_best][0]
                ptj = list(match_cands)[cand_best][1]
                if ptj == 0 or ptj == self._snakelist[snakej].NP - 1:
                    head_dist = self._snakelist[snakei][0].pos.dist(self._snakelist[snakej][ptj].pos)
                    tail_dist = self._snakelist[snakei][-1].pos.dist(self._snakelist[snakej][ptj].pos)
                    if pti == 0 and tail_dist < head_dist:
                        print('wait until tail')
                        continue
                    if DEBUG:
                        print('#',snakei, pti, snakej, ptj, 'merge')
                    if pti == 0:
                        app = False
                    else:
                        app = True
                    if ptj == 0:
                        rev = False
                    else:
                        rev = True
                    self._snakelist[snakei].mergeSnake(self._snakelist[snakej], rev, app)
                    # merged snake remove
                    snake_avail[snakej] = 0
                    if DEBUG:
                        print(snakej, 'set invalid')
                    pending_pti.append(pti)
                else:
                    if DEBUG:
                        print('#',snakei, pti, snakej, ptj, 'branch',
                          self._snakelist[snakei][pti].pos.dist(self._snakelist[snakej][ptj].pos))

                    self._snakelist[snakei].branchSnake(self._snakelist[snakej][ptj], pti)
        merge_snakelist = SnakeList()
        for snakei in range(self.NSnakes):
            if snake_avail[snakei] == 0:
                continue
            merge_snakelist.addSnake(self._snakelist[snakei])

        merge_snakelist.removeDuplicatePts()
        merge_snakelist.resampleSnakes(1)
        merge_snakelist.assignDeg()
        return merge_snakelist

    def removeDuplicatePts(self):
        for snakei in range(self.NSnakes):
            self._snakelist[snakei].removeDuplicatePts()

    def removeSelfLoop(self):
        for snakei in range(self.NSnakes):
            self._snakelist[snakei].removeSelfLoop()

    def assignDeg(self):
        deg_dict = {}
        #collect deg
        for snakei in range(self.NSnakes):
            for pti in range(self._snakelist[snakei].NP):
                ppos_has = None
                if pti>0:
                    ppos_has = self._snakelist[snakei][pti-1].pos.hashPos()
                npos_has = None
                if pti < self._snakelist[snakei].NP-1:
                    npos_has = self._snakelist[snakei][pti+1].pos.hashPos()
                if ppos_has:
                    if ppos_has in deg_dict:
                        deg_dict[ppos_has] += 1
                    else:
                        deg_dict[ppos_has] = 1
                if npos_has:
                    if npos_has in deg_dict:
                        deg_dict[npos_has] += 1
                    else:
                        deg_dict[npos_has] = 1
        #update
        for snakei in range(self.NSnakes):
            for pti in range(self._snakelist[snakei].NP):
                cpos_has = self._snakelist[snakei][pti].pos.hashPos()
                if cpos_has not in deg_dict:
                    print('cpos',cpos_has,'not in deg_dict')
                    continue
                self._snakelist[snakei][pti].type = deg_dict[cpos_has]


    def nodeGraph(self):
        G = nx.Graph()
        pt_map = {}
        for snakei in range(self.NSnakes):
            for pti in range(self._snakelist[snakei].NP):
                # add pt as node
                hash_pos = self._snakelist[snakei][pti].pos.hashPos()
                if hash_pos not in pt_map:
                    nodei = len(pt_map)
                    pt_map[hash_pos] = nodei
                else:
                    nodei = pt_map[hash_pos]
                G.add_node(nodei, pos=self._snakelist[snakei][pti].pos.lst(), rad=self._snakelist[snakei][pti].rad)

                if pti>0:
                    nodej = pt_map[self._snakelist[snakei][pti-1].pos.hashPos()]
                    cdist = self._snakelist[snakei][pti].pos.dist(self._snakelist[snakei][pti-1].pos)
                    G.add_edge(nodei,nodej,dist=cdist,snakei=snakei,pti=pti,snakej=snakei,ptj=pti-1)
        G.graph['pt_map'] = pt_map
        return G

    def branchGraph(self):
        G = nx.Graph()
        for snakei in range(self.NSnakes):
            #add branch as node
            G.add_node(snakei, NP=self._snakelist[snakei].NP, pos=self._snakelist[snakei].ct.lst())
            #add distance loss as edge
        return G


    def tree(self,icafem,search_range=30,thres_loss=1,int_src='o',DEBUG=1):
        G = nx.Graph()
        label_img = self.labelMap(icafem.tifimg.shape)
        icafem.loadImg('label_map',label_img)
        for snakei in range(self.NSnakes):
            # add branch as node
            G.add_node(snakei, NP=self._snakelist[snakei].NP, pos=self._snakelist[snakei].ct.lst())
            # add distance loss as edge
            # check head and tails 
            for pti in [0, -1]:
                cpos = self._snakelist[snakei][pti].pos
                exclude_snakeids = [snakei]
                match_cands = self.matchPts(cpos, search_range, exclude_snakeids)
                match_cands = sorted(match_cands, key = operator.itemgetter(2)) # Sort by distance -KY
                for snakej, ptj, cdist in match_cands:
                    # also ensure pti is the closest point to make the branch
                    mindist_snakei = np.inf
                    minpti = None
                    for ptii in range(self._snakelist[snakei].NP):
                        cdistii = self._snakelist[snakei][ptii].pos.dist(self._snakelist[snakej][ptj].pos)
                        if cdistii < mindist_snakei:
                            minpti = ptii
                            mindist_snakei = cdistii
                    ptir = pti if pti == 0 else self._snakelist[snakei].NP - 1
                    if minpti != ptir:
                        if DEBUG:
                            print('pti not the closest', snakei, ptir, '!=', minpti, 'remove cand', snakej, ptj,
                                  self._snakelist[snakej][ptj].pos)
                        continue
                    if cdist < 0.1:
                        closs = 0
                    else:
                        merge_snake_init = Snake()
                        merge_snake_init.add(self._snakelist[snakei][pti])
                        merge_snake_init.add(self._snakelist[snakej][ptj])
                        merge_snake_init = merge_snake_init.resampleSnake(1)
                        # merge_snake_ref = icafem.simpleRefSnake(merge_snake_init)
                        merge_snake_ref = merge_snake_init

                        label_id_in_path = icafem.getIntensityAlongSnake(merge_snake_ref, src='label_map',int_pos=True)
                        #-1 is pos covered by 2+ snake
                        label_id_in_path_exclude = [i for i in label_id_in_path if i not in [-1,snakei+1,snakej+1]]
                        if sum(label_id_in_path_exclude)!=0:
                            if DEBUG:
                                print(snakei,snakej,'labeled segment in merge path, skip',label_id_in_path_exclude)
                            continue

                        interp_int_arr = icafem.getIntensityAlongSnake(merge_snake_ref, src=int_src)
                        ori_int_arr = icafem.getIntensityAlongSnake(self._snakelist[snakei], src=int_src)
                        ori_bg_int_arr = np.array(icafem.getIntensityRaySnake(self._snakelist[snakei], src=int_src)).reshape(-1)

                        # if merge, need to compare int fit for both snake
                        if ptj in [0, -1, self._snakelist[snakej].NP - 1]:
                            target_int_arr = icafem.getIntensityAlongSnake(self._snakelist[snakej],
                                                                           src=int_src)
                            target_int_mean = np.mean(target_int_arr)
                            #print('matchness two branch', norm_distribution.cdf(target_int_mean))
                            if not icafem.mergeSnakeIntMatch(ori_int_arr,target_int_arr):
                                if DEBUG:
                                    print(snakei, snakej, 'merging target branch intensity mismatch')
                                continue
                            # if fit, merge all itnensity
                            all_int_arr = ori_int_arr + target_int_arr
                            interp_probs = icafem.pathMatchInt(all_int_arr,ori_bg_int_arr,interp_int_arr,DEBUG=DEBUG)
                        #if branch
                        else:
                            interp_probs = icafem.pathMatchInt(ori_int_arr,ori_bg_int_arr,interp_int_arr,DEBUG=DEBUG)
                        if np.min(interp_probs) < 0.05:
                            if DEBUG:
                                print(snakei, pti, 'to', snakej, ptj, 'int too low', np.min(interp_probs), 'skip')
                            continue
                        closs = 1 - np.median(interp_probs)
                    if DEBUG:
                        print(snakei, 'head' if pti == 0 else 'tail', 'match to', snakej, 'loss', closs)
                    if not G.has_edge(snakei, snakej) or (G.has_edge(snakei, snakej) and closs<G.edges[snakei,snakej]['loss']):
                        G.add_edge(snakei, snakej, dist=cdist, loss=closs, snakei=snakei, pti=pti,
                                      snakej=snakej, ptj=ptj)
                        print('G: ', snakei, 'head' if pti == 0 else 'tail', 'match to', snakej, 'loss', closs)
                    
        tree_G = nx.minimum_spanning_tree(G, 'loss')
        #plt.figure(figsize=(10, 10))
        #nx.draw_networkx(tree_G, font_size=13, node_size=[20 + tree_G.nodes[i]['NP'] for i in tree_G.nodes()],
        #                 pos={i: tree_G.nodes[i]['pos'][:2] for i in tree_G.nodes()},
        #                 node_color='r')
        #plt.show()

        #apply snakelist changes based on existing edges
        tree_snakelist = self.copy()
        for edgei in tree_G.edges():
            edge_item = tree_G.edges[edgei]
            snakei, pti, snakej, ptj = edge_item['snakei'], edge_item['pti'], edge_item['snakej'], edge_item['ptj']
            if edge_item['dist'] != 0 and edge_item['loss']<thres_loss:
                if DEBUG:
                    print('branch', snakei, pti, snakej, ptj, edge_item['dist'], edge_item['loss'])
                tree_snakelist[snakei].branchSnake(self._snakelist[snakej][ptj], pti)

        
        tree_snakelist = tree_snakelist.removeShort(3)
        tree_snakelist.removeSelfLoop()
        tree_snakelist.autoMerge()

        tree_snakelist.autoBranch()
        tree_snakelist.autoTransform(mode='length')
        tree_snakelist = tree_snakelist.removeShort(3)

        tree_snakelist = tree_snakelist.resampleSnakes(1)
        tree_snakelist.autoTransform(mode='length')
        tree_snakelist.autoBranch()
        
        #tree_snakelist = tree_snakelist.trimDuplicateSnake(icafem.shape)
        tree_snakelist.autoMerge()
        tree_snakelist.removeSelfLoop()
        
        #tree_snakelist.autoTransform()

        return tree_snakelist

    def transformSnake(self,snakei,pti,snakej,ptj,connect_reverse=False):
        assert pti in [0,-1,self._snakelist[snakei].NP-1]
        #cut snakej by ptj
        #print('bf', snakei,pti,snakej,ptj,connect_reverse,self._snakelist[snakei], self._snakelist[snakej])
        if connect_reverse == False:
            #if connect_reverse is false, merge snakei with ptj~NP-1
            cut_snake = Snake(self._snakelist[snakej].snake[ptj+1:])
            #trim snakej

            self._snakelist[snakej].trimSnake(ptj, False) # Not sure - KY  #Original is correct --LC
            print('snakei',self._snakelist[snakei],'snakej',self._snakelist[snakej],cut_snake)

            if pti == 0:
                self._snakelist[snakei].mergeSnake(cut_snake, reverse=False, append=False)
            else:
                self._snakelist[snakei].mergeSnake(cut_snake, reverse=False, append=True)
        else:
            #else, merge snakei with 0~ptj
            cut_snake = Snake(self._snakelist[snakej].snake[:ptj])
            self._snakelist[snakej].trimSnake(ptj+1, True)
            #print(cut_snake,self._snakelist[snakei],self._snakelist[snakej])
            if pti == 0:
                self._snakelist[snakei].mergeSnake(cut_snake, reverse=True, append=False)
            else:
                self._snakelist[snakei].mergeSnake(cut_snake, reverse=True, append=True)
        #print('after transform snakei',self._snakelist[snakei],'snakej',self._snakelist[snakej])

    #try to find more z range in trace
    def transformSnakeByPos(self, snakei, pti, snakej, ptj, dim=2):
        zi = [p.pos.intlst()[dim] for p in self._snakelist[snakei]]
        zj1 = [p.pos.intlst()[dim] for p in self._snakelist[snakej].snake[:ptj]]
        zj2 = [p.pos.intlst()[dim] for p in self._snakelist[snakej].snake[ptj:]]
        zj = zj1 + zj2
        connect_frontj_len = len(set(np.arange(min(zi),max(zi)).tolist()+np.arange(min(zj1),max(zj1)).tolist()))
        connect_endj_len = len(set(np.arange(min(zi),max(zi)).tolist()+np.arange(min(zj2),max(zj2)).tolist()))
        if max(len(set(np.arange(min(zi),max(zi)))),len(set(np.arange(min(zj),max(zj))))) > max(connect_frontj_len,connect_endj_len):
            return False
        #print('zj front',len(set(np.arange(min(zi),max(zi)).tolist()+np.arange(min(zj1),max(zj1)).tolist())))
        #print('zj end',len(set(np.arange(min(zi),max(zi)).tolist()+np.arange(min(zj2),max(zj2)).tolist())))
        if connect_frontj_len> connect_endj_len:
            self.transformSnake(snakei, pti, snakej, ptj, True)
        else:
            self.transformSnake(snakei, pti, snakej, ptj, False)
        return True

    def transformSnakeByLength(self, snakei, pti, snakej, ptj):
        if self._snakelist[snakei].NP < min(ptj, self._snakelist[snakej].NP-ptj):
            return False
        if ptj > self._snakelist[snakej].NP//2:
            print('transformSnakeByLength',snakei, pti, snakej, ptj)
            self.transformSnake(snakei, pti, snakej, ptj, True)
        else:
            self.transformSnake(snakei, pti, snakej, ptj, False)
        return True

    def transformSnakeByAngle(self, snakei, pti, snakej, ptj):
        assert pti in [0,-1,self._snakelist[snakei].NP-1]
        if self._snakelist[snakej][ptj].type<=2:
            print('ptj not deg 2',self._snakelist[snakej][ptj].type, 'skip')
            return
        #dir is always pos[pti+1]-pos[pti], three dir all point from birf point to distal point
        if pti==0:
            idi, idj = 0, min(self._snakelist[snakei].NP,2)
        else:
            idi, idj = -1, max(0,-2)
        single_dir = (self._snakelist[snakei][idj].pos-self._snakelist[snakei][idi].pos).norm()
        idi, idj = ptj, max(0, ptj-2)
        branch_dir1 = (self._snakelist[snakej][idj].pos-self._snakelist[snakej][idi].pos).norm()
        idi, idj = ptj, min(self._snakelist[snakej].NP-1, ptj+2)
        branch_dir2 = (self._snakelist[snakej][idj].pos-self._snakelist[snakej][idi].pos).norm()
        vecs1 = single_dir.prod(branch_dir1)
        vecs2 = single_dir.prod(branch_dir2)
        vec12 = branch_dir1.prod(branch_dir2)
        minvec = min(vec12,vecs1,vecs2)
        #ideal should be -1
        print('minvec',minvec,'from',vecs1,vecs2,vec12)
        if minvec == vecs1:
            print('single branch connect to prior segment of double branch')
            self.transformSnake(snakei, pti, snakej, ptj, True)
            return 1
        elif minvec==vecs2:
            print('single branch connect to later segment of double branch')
            self.transformSnake(snakei, pti, snakej, ptj, False)
            return 1
        else:
            print('no change needed')
            return 0
            #no need to change

    def autoTransform(self,mode='angle',thres=3,pos_dim=2,DEBUG=0):
        checked_pos = set()

        for snakei in range(self.NSnakes):
            if self._snakelist[snakei].NP<=thres:
                continue
            #transform need iterative process, new head/tail need to search again
            # check head and tail
            ptchecklist = [0,-1]
            while len(ptchecklist):
                ckpt = ptchecklist.pop(0)
                #print('check',snakei,ckpt)
                cpos = self._snakelist[snakei][ckpt].pos
                #crad = self._snakelist[snakei][ckpt].rad
                match_snakeid, match_ptid, match_dist, match_rad = self.matchPt(cpos, snakei)
                # only for deg 3+ pts
                if match_dist == 0:
                    if DEBUG:
                        print('ckt', snakei, ckpt, match_snakeid, match_ptid)
                    if self._snakelist[match_snakeid][match_ptid].pos.hashPos() in checked_pos:
                        if DEBUG:
                            print('checked')
                        continue
                    else:
                        checked_pos.add(self._snakelist[match_snakeid][match_ptid].pos.hashPos())
                    if self._snakelist[match_snakeid].NP <= 3:
                        continue
                    if match_ptid==0 or match_ptid == self._snakelist[match_snakeid].NP-1:
                        continue
                    #print('transform', snakei, ckpt, match_snakeid, match_ptid)
                    ptchecklist.append(ckpt)
                    if ckpt == -1:
                        ckpt = self._snakelist[snakei].NP - 1
                    if mode=='angle':
                        transformed = self.transformSnakeByAngle(snakei, ckpt, match_snakeid, match_ptid)
                    elif mode == 'length':
                        transformed = self.transformSnakeByLength(snakei, ckpt, match_snakeid, match_ptid)
                    elif mode == 'pos':
                        transformed = self.transformSnakeByPos(snakei, ckpt, match_snakeid, match_ptid, pos_dim)
                    else:
                        raise ValueError('Type err')

    def mainArtTree(self,dist_thres=20,minbirf=5):
        def locProb(c,snakelist_ct_mean,snakelist_ct_std):
            loc = [c.nodes[ci]['pos'].lst() for ci in c.nodes()]
            loc_ct = np.mean(loc,axis=0)
            x_norm_dist = scipy.stats.norm(snakelist_ct_mean[0], snakelist_ct_std[0])
            y_norm_dist = scipy.stats.norm(snakelist_ct_mean[1], snakelist_ct_std[1])
            x_pos = loc_ct[0]
            y_pos = loc_ct[1]

            x_prob = min(x_norm_dist.cdf(x_pos), 1 - x_norm_dist.cdf(x_pos))
            y_prob = min(y_norm_dist.cdf(y_pos), 1 - x_norm_dist.cdf(y_pos))
            return min(x_prob, y_prob)


        snake_cts = []
        G = nx.Graph()
        for snakei in range(self.NSnakes):
            csnake_ct = self._snakelist[snakei].ct
            snake_cts.append(csnake_ct.lst())
            G.add_node(snakei,NP=self._snakelist[snakei].NP,pos=csnake_ct)
            #for pti in [0,-1]:
            for pti in range(self._snakelist[snakei].NP):
                cpos = self._snakelist[snakei][pti].pos
                match_snake_ids = self.matchPts(cpos, dist_thres)
                for match_snake_id in match_snake_ids:
                    target_snakeid = match_snake_id[0]
                    if snakei==target_snakeid:
                        continue
                    G.add_edge(snakei,target_snakeid)
        snakelist_ct_mean = np.mean(snake_cts,axis=0)
        snakelist_ct_std = np.std(snake_cts,axis=0)
        node_num_part = [len(G.subgraph(c).copy().nodes()) for c in nx.connected_components(G)]
        print(node_num_part)
        if minbirf>=np.max(node_num_part):
            minbirf = np.max(node_num_part)-1
        valid_components = [G.subgraph(c).copy() for c in nx.connected_components(G) \
                            if len(G.subgraph(c).copy().nodes())>minbirf and \
                            locProb(G.subgraph(c),snakelist_ct_mean,snakelist_ct_std)>0.05]
        print(len(valid_components),'valid_components, largest',len(valid_components[0].nodes()))
        if len(valid_components)==0:
            print('no enough pt in graph')
            valid_graph = max(nx.connected_components(G), key=len)
        elif len(valid_components) == 1:
            valid_graph = valid_components[0]
        else:
            valid_graph = valid_components[0]
            for c in range(1,len(valid_components)):
                valid_graph = nx.compose(valid_components[c], valid_graph)
        #print(valid_graph.nodes())
        #add snake to main snakelist
        main_snakelist = SnakeList()
        for snakei in valid_graph.nodes():
            main_snakelist.addSnake(self._snakelist[snakei])
        return main_snakelist

    #paint zeros stack with snakeid, ptid tuple for artery centerline
    def idMap(self,box):
        target_shape = list(box)+[2]
        id_map = np.ones(target_shape,dtype=np.int)*(-1)
        for snakei in range(self.NSnakes):
            for pti in range(self._snakelist[snakei].NP):
                cpos = self._snakelist[snakei][pti].pos
                if id_map[tuple(cpos.intlst())][0] in [snakei,-1]:
                    id_map[tuple(cpos.intlst())] = [snakei,pti]
                else:
                    print('snake',snakei,'pti',pti,'at',cpos.intlst(),'has conflict with previous snake',id_map[tuple(cpos.intlst())])
        return id_map

    #paint zeros stack with id for artery region within radius
    def labelMap(self,shape):
        label_img = np.zeros(shape, dtype=np.int16)
        for snakeid in range(self.NSnakes):
            # print('\rpainting snake',snakeid,end='')
            for pti in range(self._snakelist[snakeid].NP):
                pos = self._snakelist[snakeid][pti].pos
                rad = self._snakelist[snakeid][pti].rad
                # paint within radius at pos position
                paint_dist_unique(label_img, pos, rad, snakeid + 1)
        return label_img

    def toTraceList(self):
        traces = []
        for snakei in range(self.NSnakes):
            clist = []
            for pti in range(self._snakelist[snakei].NP):
                clist.append(np.array(self._snakelist[snakei][pti].pos.intlst()))
            traces.append(clist)
        return traces

    #find a pt in snakelist nearest to a position
    def matchPt(self,pos,exclude_snakeid=None,thres_rad=5):
        match_snakeid, match_ptid, match_dist, match_rad = -1, -1, np.inf, -1
        for snakei in range(self.NSnakes):
            if exclude_snakeid is not None and snakei == exclude_snakeid:
                continue
            if pos.outOfBox(self._snakelist[snakei].box,thres_rad):
                continue
            for pti in range(self._snakelist[snakei].NP):
                cdist = pos.dist(self._snakelist[snakei][pti].pos)
                if cdist<match_dist and cdist<thres_rad:
                    match_dist = cdist
                    match_snakeid = snakei
                    match_ptid = pti
                    match_rad = self._snakelist[snakei][pti].rad
        return match_snakeid, match_ptid, match_dist, match_rad

    #return all points (snakei and pti) within thres rad of pos
    def matchPts(self,pos,thres_rad=5,exclude_snakeids=[]):
        #match info with mindist for each snakei, avoid pts on the same snake counted multiple times
        match_ids = {}
        for snakei in range(self.NSnakes):
            if snakei in exclude_snakeids:
                continue
            if pos.outOfBox(self._snakelist[snakei].box,thres_rad):
                continue
            for pti in range(self._snakelist[snakei].NP):
                if self._snakelist[snakei][pti].link_id is not None:
                    continue
                cdist = pos.dist(self._snakelist[snakei][pti].pos)
                if cdist<thres_rad:
                    if snakei not in match_ids:
                        match_ids[snakei] = (snakei,pti,cdist)
                    else:
                        if cdist<match_ids[snakei][2]:
                            match_ids[snakei] = (snakei,pti,cdist)
        return match_ids.values()

    def compRefSnakelist(self,ref_snakelist):
        self.resampleSnakes(1)
        ref_snakelist.resampleSnakes(1)
        self.comp_map = []
        #comp from snake
        for snakei in range(self.NSnakes):
            comp_snake_map = []
            for pti in range(self._snakelist[snakei].NP):
                #each traced point match to reference snakelist
                match_snakeid, match_ptid, match_dist, match_rad = ref_snakelist.matchPt(self._snakelist[snakei][pti].pos)
                if match_dist<5:
                    comp_snake_map.append((match_snakeid, match_ptid, match_dist, match_rad))
                else:
                    comp_snake_map.append((-1,-1,match_dist))
            self.comp_map.append(comp_snake_map)

        #comp from ref
        self.ref_comp_map = []
        for snakei in range(ref_snakelist.NSnakes):
            ref_comp_snake_map = []
            for pti in range(ref_snakelist[snakei].NP):
                # each reference point match to traced snakelist
                match_snakeid, match_ptid, match_dist, match_rad = self.matchPt(ref_snakelist[snakei][pti].pos)
                #replace gt rad. from matchPt function the rad is traced artery, not what we want
                match_rad = ref_snakelist[snakei][pti].rad
                if match_dist < 5:
                    ref_comp_snake_map.append((match_snakeid, match_ptid, match_dist, match_rad))
                else:
                    ref_comp_snake_map.append((-1, -1, match_dist))
            self.ref_comp_map.append(ref_comp_snake_map)

    def evalCompDist(self):
        if self.comp_map is None:
            print('compRefSnakelist needed')
            return
        #points on traced artery are within the ref radius
        TPM_ov = 0
        TPM_ot = 0
        #points on ref artery has matching points from traced artery
        TPR_ov = 0
        TPR_of = 0
        TPR_ot = 0
        #only TP points calculate diffs
        diffs = []
        ref_diffs = []

        #matched but distance larger than ref radius
        FP_ov = 0
        FP_ot = 0
        self.FPlist = []

        FN_ov = 0
        FN_of = 0
        FN_ot = 0
        #traced but unmatch point
        UM = 0
        #traced but unmatch snakes
        UMS = 0
        # reference points but unmatch
        ref_UM = 0
        # reference but unmatch snakes
        ref_UMS = 0


        #eval on traced arteries
        for snakei in range(len(self.comp_map)):
            if [pti[0] for pti in self.comp_map[snakei]].count(-1) / len(self.comp_map[snakei])>0.5:
                UM += len(self.comp_map[snakei])
                UMS += 1
                FP_ov += len(self.comp_map[snakei])
                FP_ot += len(self.comp_map[snakei])
                continue
            made_first_error = False
            radius_below_thres = False
            for pti in range(len(self.comp_map[snakei])):
                if self.comp_map[snakei][pti][0]==-1:
                    UM += 1
                    FP_ov += 1
                else:
                    if radius_below_thres == False and self.comp_map[snakei][pti][3]<0.75/0.4297:
                        radius_below_thres = True

                    #difference smaller than radius
                    if self.comp_map[snakei][pti][2]<self.comp_map[snakei][pti][3]:
                        TPM_ov += 1
                        diffs.append(self.comp_map[snakei][pti][2])
                        if radius_below_thres == False:
                            TPM_ot += 1
                    else:
                        FP_ov += 1
                        self.FPlist.append(self.comp_map[snakei][pti])
                        if made_first_error == False:
                            made_first_error = True
                        if radius_below_thres == False:
                            FP_ot += 1

        # eval on reference arteries
        for snakei in range(len(self.ref_comp_map)):
            if [pti[0] for pti in self.ref_comp_map[snakei]].count(-1) / len(self.ref_comp_map[snakei]) > 0.5:
                ref_UM += len(self.ref_comp_map[snakei])
                ref_UMS += 1
                FN_ov += len(self.ref_comp_map[snakei])
                FN_ot += len(self.ref_comp_map[snakei])
                FN_of += len(self.ref_comp_map[snakei])
                continue
            made_first_error = False
            radius_below_thres = False
            for pti in range(len(self.ref_comp_map[snakei])):
                if self.ref_comp_map[snakei][pti][0] == -1:
                    ref_UM += 1
                    FN_ov += 1
                else:
                    if radius_below_thres == False and self.ref_comp_map[snakei][pti][3] < 0.75 / 0.4297:
                        radius_below_thres = True

                    # difference smaller than radius
                    if self.ref_comp_map[snakei][pti][2] < self.ref_comp_map[snakei][pti][3]:
                        TPR_ov += 1
                        ref_diffs.append(self.ref_comp_map[snakei][pti][2])
                        if radius_below_thres == False:
                            TPR_ot += 1
                        if made_first_error == False:
                            TPR_of += 1
                    else:
                        FN_ov += 1
                        self.FPlist.append(self.ref_comp_map[snakei][pti])
                        if made_first_error == False:
                            made_first_error = True
                        if radius_below_thres == False:
                            FN_ot += 1
                    if made_first_error == True:
                        FN_of += 1
        OV = (TPM_ov+TPR_ov)/(TPM_ov+TPR_ov+FN_ov+FP_ov)
        OF = TPR_of/(TPR_of+FN_of)
        OT = (TPM_ot+TPR_ot)/(TPM_ot+TPR_ot+FN_ot+FP_ot)
        AI = np.mean(ref_diffs)
        print('OV (Overlap): %.3f'%OV)
        print('OF (Overlap until first error): %.3f' % OF)
        print('OT (Overlap with the clinically relevant part of the vessel): %.3f' % OT)
        print('AI (Average inside):%.3f'%AI)
        print('TPM_ov:%d,TPR_ov:%d,FN_ov:%d,FP_ov:%d'%(TPM_ov,TPR_ov,FN_ov,FP_ov))
        print('TPR_of:%d,FN_of:%d'%(TPR_of,FN_of))
        print('TPM_ot:%d,TPR_ot:%d,FN_ot:%d,FP_ot:%d'%(TPM_ot,TPR_ot,FN_ot,FP_ot))
        return OV, OF, OT, AI, UM, UMS, ref_UM, ref_UMS, np.mean(diffs)

    def resetLinkID(self):
        for snakei in range(self.NSnakes):
            for pti in range(self._snakelist[snakei].NP):
                self._snakelist[snakei][pti].link_id = None

    def motMetric(self,veslist,type='all'):
        self.resetLinkID()
        all_metric = MOTMetric()
        #proximal_metric = MOTMetric()
        #distal_metric = MOTMetric()

        MAJOR_BRANCH = [1, 2, 3, 4, 7, 8, 14, 15, 16]

        ves_snakelist = SnakeList()
        for ves_type in range(len(veslist)):
            for vesi in range(len(veslist[ves_type])):
                #print('ves', ves_type, 'vesi', vesi)
                cves_snake = veslist[ves_type][vesi].resampleSnake(1)
                cves_snake.id = ves_snakelist.NSnakes
                snake_metric = cves_snake.matchComp(self)
                ves_snakelist.addSnake(cves_snake)
                all_metric.addSnakeMetric(snake_metric)
                '''if ves_type in MAJOR_BRANCH:
                    proximal_metric.addSnakeMetric(snake_metric)
                else:
                    distal_metric.addSnakeMetric(snake_metric)'''

        all_metric.setSnakelist(ves_snakelist,self)
        #proximal_metric.setSnakelist(ves_snakelist,self)
        #distal_metric.setSnakelist(ves_snakelist,self)

        return all_metric


    def feats(self,feat_names=None):
        feat_funcs = {'artery number':self.NSnakes,'length':self.length,'volume':self.volume,'tortuosity':self.tot,
                      'proximal artery number':self.branchProximal,'distal artery number':self.branchDistal,
                      'proximal length':self.lengthProximal,'distal length':self.lengthDistal, 
                      'proximal volume':self.volumeProximal,'distal volume':self.volumeDistal,
                      'proximal tortuosity': self.totProximal, 'distal tortuosity': self.totDistal,
                      'left artery number':self.branchLeft, 'right artery number':self.branchRight,
                      'left length':self.lengthLeft, 'right length':self.lengthRight,
                      'left volume':self.volumeLeft, 'right volume':self.volumeRight,
                      'left tortuosity': self.totLeft, 'right tortuosity': self.totRight,
                      'distal ACA artery number': self.branchDistalACA,
                      'distal ACA length': self.lengthDistalACA,
                      'distal ACA volume': self.volumeDistalACA,
                      'distal ACA tortousity': self.totDistalACA,
                      'distal MCA artery number': self.branchDistalMCA,
                      'distal MCA length': self.lengthDistalMCA,
                      'distal MCA volume': self.volumeDistalMCA,
                      'distal MCA tortousity': self.totDistalMCA,
                      'distal PCA artery number': self.branchDistalPCA,
                      'distal PCA length': self.lengthDistalPCA,
                      'distal PCA volume': self.volumeDistalPCA,
                      'distal PCA tortousity': self.totDistalPCA}
        if feat_names is None:
            feat_names = feat_funcs.keys()
        feats = {}
        for feat_name in feat_names:
            feats[feat_name] = feat_funcs[feat_name]
        return feats

    def featsToMM(self,feats,res):
        feat_mm = {'artery number': 1, 'length': res, 'volume': res**3, 'tortuosity': 1,
                      'proximal artery number': 1, 'distal artery number': 1, 
                      'proximal length': res, 'distal length': res, 
                      'proximal volume': res**3, 'distal volume': res**3, 
                      'proximal tortuosity': 1, 'distal tortuosity': 1,
                      'left artery number':1, 'right artery number':1,
                      'left length':res, 'right length':res,
                      'left volume':res**3, 'right volume':res**3,
                      'left tortuosity': 1, 'right tortuosity': 1,
                      'distal ACA artery number': 1, 'distal ACA length': res, 'distal ACA volume': res**3, 'distal ACA tortousity': 1,
                      'distal MCA artery number': 1, 'distal MCA length': res, 'distal MCA volume': res**3, 'distal MCA tortousity': 1,
                      'distal PCA artery number': 1, 'distal PCA length': res, 'distal PCA volume': res**3, 'distal PCA tortousity': 1}
        feats_in_mm = {}
        for feat_name in feats:
            feats_in_mm[feat_name] = feats[feat_name]*feat_mm[feat_name]
        return feats_in_mm

    def printFeats(self,feats):
        for feat_name in feats:
            print(feat_name,'\t','%.2f'%feats[feat_name])

    def printFeatsMM(self,feats):
        feat_unit = {'artery number': '', 'length': 'mm', 'volume': 'mm^3', 'tortuosity': '',
                   'proximal artery number': '', 'distal artery number': '', 
                   'proximal length': 'mm', 'distal length': 'mm', 
                   'proximal volume': 'mm^3', 'distal volume': 'mm^3',
                   'proximal tortuosity': '', 'distal tortuosity': '',
                   'left artery number':'', 'right artery number':'',
                   'left length':'mm', 'right length':'mm',
                   'left volume':'mm^3', 'right volume':'mm^3',
                   'left tortuosity': '', 'right tortuosity': '',
                   'distal ACA artery number': '', 'distal ACA length': 'mm', 'distal ACA volume': 'mm^3', 'distal ACA tortousity': '',
                   'distal MCA artery number': '', 'distal MCA length': 'mm', 'distal MCA volume': 'mm^3', 'distal MCA tortousity': '',
                   'distal PCA artery number': '', 'distal PCA length': 'mm', 'distal PCA volume': 'mm^3', 'distal PCA tortousity': ''}
        for feat_name in feats:
            if type(feats[feat_name])==int:
                print(feat_name,'\t','%d'%feats[feat_name],feat_unit[feat_name])
            else:
                print(feat_name,'\t','%.2f'%feats[feat_name],feat_unit[feat_name])

    def eval_simple(self,ref_snakelist):
        snakelist = self.copy()
        _ = snakelist.resampleSnakes(1)
        # ground truth snakelist from icafem.veslist
        all_metic = snakelist.motMetric(ref_snakelist)
        metric_dict = all_metic.metrics(['MOTA', 'IDF1', 'MOTP', 'IDS'])
        snakelist.compRefSnakelist(ref_snakelist)
        metric_dict['OV'] = snakelist.evalCompDist()[0]
        str = ''
        for key in metric_dict:
            str += key + '\t'
        str += '\n'
        for key in metric_dict:
            if type(metric_dict[key]) == int:
                str += '%d\t' % metric_dict[key]
            else:
                str += '%.3f\t' % metric_dict[key]
        print(str)


    def trimDuplicateSnake(self, shape, snake_order=None):
        if snake_order is None:
            snake_npts = [self._snakelist[i].box_vol for i in range(self.NSnakes)]
            snake_order = np.argsort(snake_npts)[::-1]

        IDMap = np.ones(shape, dtype=np.int) * (-1)
        valid_snakelist = SnakeList()
        for snakei in snake_order:
            csnake = Snake()
            last_hit = None
            paint_pos = []
            for pti in range(self._snakelist[snakei].NP):
                int_pos = self._snakelist[snakei][pti].pos.intlst()
                if int_pos[0] >= shape[0]:
                    int_pos[0] = shape[0] - 1
                if int_pos[1]>=shape[1]:
                    int_pos[1] = shape[1] - 1
                if int_pos[2]>=shape[2]:
                    int_pos[2] = shape[2] - 1
                c_hit = IDMap[tuple(int_pos)]
                if c_hit in [-1]:
                    csnake.add(self._snakelist[snakei][pti])
                    paint_pos.append(self._snakelist[snakei][pti])
                else:
                    if csnake.NP > 1:
                        if last_hit is not None:
                            # add first pointn if nearby
                            branch_pt = self._snakelist[last_hit].findBranchPt(csnake[0].pos)
                            if branch_pt.pos.dist(csnake[0].pos)<5:
                                csnake.insert(0, branch_pt)
                        # add last point
                        branch_pt = self._snakelist[c_hit].findBranchPt(csnake[-1].pos)
                        if branch_pt.pos.dist(csnake[-1].pos) < 5:
                            csnake.add(branch_pt)
                        valid_snakelist.addSnake(csnake)
                        last_hit = None
                        csnake = Snake()
                    else:
                        last_hit = c_hit
            if csnake.NP > 1:
                if last_hit is not None:
                    branch_pt = self._snakelist[last_hit].findBranchPt(csnake[0].pos)
                    if branch_pt.pos.dist(csnake[0].pos) < 5:
                        csnake.insert(0, branch_pt)
                valid_snakelist.addSnake(csnake)
            # paint
            for posi in paint_pos:
                paintBall(IDMap, posi, snakei)
        plt.imshow(np.max(IDMap, axis=2))
        plt.show()
        return valid_snakelist

    def removeDuplicateSnake(self, box, snake_order=None):
        #order to paint
        if snake_order is None:
            snake_npts = [self._snakelist[i].box_vol for i in range(self.NSnakes)]

            snake_order = np.argsort(snake_npts)[::-1]

        IDMap = np.ones(box, dtype=np.int) * (-1)
        PtIDMap = np.ones(box, dtype=np.int) * (-1)
        valid_snakelist = SnakeList()
        for snakei in snake_order:
            csnake = Snake()
            last_hit = None

            for pti in range(self._snakelist[snakei].NP):
                int_pos = self._snakelist[snakei][pti].pos.intlst()
                if int_pos[0] >= box[0]:
                    int_pos[0] = box[0] - 1
                if int_pos[1]>=box[1]:
                    int_pos[1] = box[1]-1
                if int_pos[2]>=box[2]:
                    int_pos[2] = box[2]-1
                c_hit = IDMap[tuple(int_pos)]
                cpt_hit = PtIDMap[tuple(int_pos)]
                if c_hit in [-1] or c_hit==snakei and (pti-cpt_hit)<5:
                    csnake.add(self._snakelist[snakei][pti])
                    posi = self._snakelist[snakei][pti]
                    paintBall(IDMap, posi, snakei)
                    paintBall(PtIDMap, posi, pti)
                else:
                    if csnake.NP > 1:
                        if last_hit is not None:
                            # add first point
                            branch_pt = self._snakelist[last_hit].findBranchPt(csnake[0].pos)
                            csnake.insert(0, branch_pt)
                        # add last point
                        branch_pt = self._snakelist[c_hit].findBranchPt(csnake[-1].pos)
                        csnake.add(branch_pt)
                        valid_snakelist.addSnake(csnake)
                        last_hit = None
                        csnake = Snake()
                    else:
                        last_hit = c_hit
            if csnake.NP > 1:
                if last_hit is not None:
                    branch_pt = self._snakelist[last_hit].findBranchPt(csnake[0].pos)
                    csnake.insert(0, branch_pt)
                valid_snakelist.addSnake(csnake)

        #plt.imshow(np.max(IDMap, axis=2))
        #plt.show()
        return valid_snakelist

    def ptAt(self,z):
        pos_list = []
        for snakei in range(self.NSnakes):
            for pti in range(self._snakelist[snakei].NP):
                if abs(self._snakelist[snakei][pti].pos.z-z)<1:
                    pos_list.append([self._snakelist[snakei][pti].pos,snakei,pti])
        return pos_list

    def cowComplete(self, res):
        anterior_circ = {'ICA_L': [0, 1], 'ICA_R': [0, 2], 'A1_L': [0, 7], 'A1_R': [0, 8], 'A2_L': [0, 9], 'A2_R': [0, 10], 'AComm': [0, 11]}
        posterior_circ = np.zeros(6)
        # artery type can be found in definition.py
        for snake in self._snakelist:
            if snake.type == 1:
                anterior_circ['ICA_L'][0] = 1
            elif snake.type == 2:
                anterior_circ['ICA_R'][0] = 1
            elif snake.type == 7:
                anterior_circ['A1_L'][0] = 1
            elif snake.type == 8:
                anterior_circ['A1_R'][0] = 1
            elif snake.type == 9:
                anterior_circ['A2_L'][0]= 1
            elif snake.type == 10:
                anterior_circ['A2_R'][0] = 1
            elif snake.type == 11:
                if np.min(snake.radList) * res * 2 > 1: 
                    anterior_circ['AComm'][0] = 1
                else:
                    print('hypoplasia of AComm')
            elif snake.type == 17:
                posterior_circ[0] = 1
            elif snake.type == 18:
                posterior_circ[1] = 1
            elif snake.type == 19:
                posterior_circ[2] = 1
            elif snake.type == 20:
                posterior_circ[3] = 1
            elif snake.type == 21:
                if np.mean(snake.radList) * res * 2 > 1: 
                    posterior_circ[4] = 1
                else:
                    print('hypoplasia of artery 21')
            elif snake.type == 22:
                if np.mean(snake.radList) * res * 2 > 1: 
                    posterior_circ[5] = 1 
                else:
                    print('hypoplasia of artery 22')
        if np.sum(np.array(list(anterior_circ.values())), axis = 0)[0] == 7 and sum(posterior_circ) < 6:
            print('partial complete with complete anterior circulation!')
        elif sum(posterior_circ) == 6 and np.sum(np.array(list(anterior_circ.values())), axis = 0)[0] < 7:
            print('partial complete with complete posterior circulation!')
        elif np.sum(np.array(list(anterior_circ.values())), axis = 0)[0] == 7 and sum(posterior_circ) == 6:
            print('complete circle of Willis!')
        else:
            print('Incomplete circle of Willis!')
        return anterior_circ, posterior_circ

    def removeSelfOverlaps(self,shape):
        break_snakelist = SnakeList()
        for snakei in range(self.NSnakes):
            print('\rremove overlap snakei', snakei, end='')
            break_pts = self._snakelist[snakei].removeSelfOverlap()
            if len(break_pts) == 2:
                break_snakelist.addSnake(self._snakelist[snakei])
            for i in range(1, len(break_pts)):
                break_snakelist.addSnake(self._snakelist[snakei].subSnake(break_pts[i - 1], break_pts[i]))
        valid_snakelist = break_snakelist.trimDuplicateSnake(shape)
        return valid_snakelist

