import math
import numpy as np
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ..point3d import Point3D

#select tracklets from all candidates using k-mean. k=2 for carotid
def kmean_remove(snakelist, thres=25, k = 2, useartcent = 1,boundary1=0,boundary2=np.inf):
    seqlabel = [0]*snakelist.NSnakes

    if snakelist.NSnakes<3:
        print('no need remove')
        return seqlabel

    xypos = []
    for snakei in range(snakelist.NSnakes):
        for pti in range(snakelist[snakei].NP):
            cpos = snakelist[snakei][pti].pos
            if cpos.z<boundary1 or cpos.z>boundary2:
                continue
            xypos.append([cpos.x,cpos.y,cpos.z])

    kmeans = KMeans(n_clusters=k, random_state=0).fit(xypos)

    kcenter_arr = kmeans.cluster_centers_
    kcenter = [Point3D(i) for i in kcenter_arr]
    print('kmean center',kcenter)
    plt.scatter([xyposi[0] for xyposi in xypos],[xyposi[1] for xyposi in xypos])
    plt.scatter([kcenter_arr[0][0],kcenter_arr[1][0]], [kcenter_arr[0][1],kcenter_arr[1][1]])


    if useartcent:
        matchart = [-1] * k
        close_by_snakes = []
        # find closest snake to each cluster center
        for arti in range(k):
            mindist = np.inf
            minsnakeid = None
            for snakei in range(snakelist.NSnakes):
                if snakei in matchart or snakei in close_by_snakes:
                    continue
                for pti in range(snakelist[snakei].NP):
                    cpos = snakelist[snakei][pti].pos
                    # minimum art distance to cluster center, divide by art length
                    ncdist = cpos.dist(kcenter[arti])/snakelist[snakei].NP
                    if ncdist<mindist:
                        mindist = ncdist
                        minsnakeid = snakei
            if mindist < thres:
                matchart[arti] = minsnakeid
                # also mark all neighbor traces of snakei as invalid
                for snakej in range(snakelist.NSnakes):
                    if snakej == minsnakeid:
                        continue
                    if snakelist[snakej].dist(snakelist[minsnakeid]) < thres or \
                            snakelist[snakej].ct.dist2d(snakelist[minsnakeid].ct)<thres:
                        close_by_snakes.append(snakej)
                        print(minsnakeid,'add neighbor', snakej)

        if len(matchart)!=len(set(matchart)):
            print('cluster fail',matchart)
            return
        else:
            print('match art id',matchart)
            xypos_match = []
            for snakei in matchart:
                for pti in range(snakelist[snakei].NP):
                    cpos = snakelist[snakei][pti].pos
                    xypos_match.append([cpos.x, cpos.y, cpos.z])
            plt.scatter([xyposi[0] for xyposi in xypos_match], [xyposi[1] for xyposi in xypos_match], c='r')
            plt.show()
            #for ki in range(k):
            #    kcenter[ki].z = np.mean([snakelist[matchart[ki]][i].pos.z for i in range(snakelist[matchart[ki]].NP)])

        kmean_center_snakeid = copy.copy(matchart)
        #matchart: queue for next checking snake
        #matchedart: saves all valid results
        matchedart = copy.copy(matchart)

        #breadth first search for all neighboring branches
        while len(matchart)>0:
            matchid = matchart.pop()
            for snakei in range(snakelist.NSnakes):
                if snakei in matchedart or snakei == matchid:
                    continue
                for spti in np.arange(0,snakelist[snakei].NP,5).tolist()+[-1]:
                    posi = snakelist[snakei][spti].pos
                    #for each neighboring branch, test its head/tail pt to any point on the match snake
                    for pti in range(snakelist[matchid].NP):
                        cdist = posi.dist(snakelist[matchid][pti].pos)
                        if cdist<thres:
                            print('head/tail point', snakei, 'mindist', cdist, matchid)
                            if snakei not in matchedart:
                                matchedart.append(snakei)
                                matchart.append(snakei)
                            break
                    if snakei in matchedart:
                        break


    else:
        matchedart = []
        for ki in range(k):
            match_snakes = snakelist.matchPts(kcenter[ki], exclude_snakeids=[], thres_rad=thres)
            for snakei, pti, cdist in match_snakes:
                if snakei not in matchedart:
                    matchedart.append(snakei)

    print('matchedart',matchedart)
    # label unmatched to -1, kmean center snakeid is 0, other is 1
    for snakei in range(snakelist.NSnakes):
        if snakei not in matchedart:
            seqlabel[snakei] = -1
        elif snakei in kmean_center_snakeid:
            seqlabel[snakei] = 0
        else:
            seqlabel[snakei] = 1

    for snakei in range(snakelist.NSnakes):
        #print(seqi,len(seqbb[seqi]))
        sliceq = [snakelist[snakei][slicei].pos.z for slicei in range(snakelist[snakei].NP)]
        xseq = [snakelist[snakei][slicei].pos.x  for slicei in range(snakelist[snakei].NP)]
        if seqlabel[snakei] == -1:
            plt.plot(xseq, sliceq, 'o', color='k', label='Trace%d' % snakei)
        elif seqlabel[snakei] == 0:
            plt.plot(xseq, sliceq, 'o', color='r', label='Trace%d' % snakei)
        else:
            plt.plot(xseq, sliceq, 'o', color='b', label='Trace%d' % snakei)
    #pyplot.legend(loc="lower right")
    #pyplot.xlim([200,550])
    plt.ylabel('slice number of each bounding box')
    plt.xlabel('x position of each bounding box')
    plt.gca().invert_yaxis()
    plt.show()

    return seqlabel
