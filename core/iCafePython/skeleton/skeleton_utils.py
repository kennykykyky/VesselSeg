import numpy as np
from ..point3d import Point3D
import skimage.graph
import matplotlib.pyplot as plt

def dist(pt1,pt2):
    return np.sqrt(np.sum([pow((pt1[dim] - pt2[dim]), 2) for dim in range(len(pt1))]))


def findRad(pti, simg):
    MAX_RAD = 10
    for rad in range(2, MAX_RAD):
        for ofi in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]:
            ptx = int(round(pti[0] + rad * ofi[0]))
            pty = int(round(pti[1] + rad * ofi[1]))
            ptz = int(round(pti[2] + rad * ofi[2]))
            if Point3D(ptx, pty, ptz).outOfBound(simg.shape):
                return rad
            if simg[ptx, pty, ptz] != True:
                return rad
    return rad


# find min index from id Map
def minInd(cpos, idMap, exclude_id):
    rad = 10
    xmin = int(np.floor(max(0, cpos[0] - rad)))
    xmax = int(np.ceil(min(idMap.shape[0] - 1, cpos[0] + rad)))
    ymin = int(np.floor(max(0, cpos[1] - rad)))
    ymax = int(np.ceil(min(idMap.shape[1] - 1, cpos[1] + rad)))
    zmin = int(np.floor(max(0, cpos[2] - rad)))
    zmax = int(np.ceil(min(idMap.shape[2] - 1, cpos[2] + rad)))

    mindist = np.inf
    minid = (None, None)
    for xi in range(xmin, xmax):
        for yi in range(ymin, ymax):
            for zi in range(zmin, zmax):
                if idMap[xi, yi, zi, 0] not in [exclude_id, -1]:
                    cdist = Point3D(xi, yi, zi).dist(Point3D(cpos))
                    if cdist < mindist:
                        mindist = cdist
                        minid = idMap[xi, yi, zi]
    if mindist > 10:
        minid = (None, None)
    return mindist, minid


# refresh idMap
def repaint(idMap, all_traces, tracei):
    for pti in range(len(all_traces[tracei])):
        x = int(np.round(all_traces[tracei][pti][0]))
        y = int(np.round(all_traces[tracei][pti][1]))
        z = int(np.round(all_traces[tracei][pti][2]))
        idMap[x,y,z] = [tracei, pti]


def findPath(simg,box, pos_s, pos_e):
    bd = 10
    xmin = int(round(max(0, min(pos_s[0], pos_e[0]) - bd)))
    ymin = int(round(max(0, min(pos_s[1], pos_e[1]) - bd)))
    zmin = int(round(max(0, min(pos_s[2], pos_e[2]) - bd)))
    xmax = int(round(min(box[0], max(pos_s[0], pos_e[0]) + bd + 1)))
    ymax = int(round(min(box[1], max(pos_s[1], pos_e[1]) + bd + 1)))
    zmax = int(round(min(box[2], max(pos_s[2], pos_e[2]) + bd + 1)))

    seg_prob_patch = simg[xmin:xmax, ymin:ymax, zmin:zmax]
    # print(xmin,ymin,zmin,xmax,ymax,zmax,seg_prob_patch.shape)
    seg_prob_patch[seg_prob_patch > 1] = 1
    seg_prob_patch = 1 - seg_prob_patch

    start_off = tuple(pos_s - np.array([xmin, ymin, zmin]))
    end_off = tuple(pos_e - np.array([xmin, ymin, zmin]))
    # print(seg_prob_patch.shape,start_off,end_off)
    path, cost = skimage.graph.route_through_array(
        seg_prob_patch, start=start_off, end=end_off, fully_connected=True)

    if cost / len(path) < 0.5:
        # print(cost,len(path))
        return [np.array(pi) + np.array([xmin, ymin, zmin]) for pi in path]
    else:
        return []




