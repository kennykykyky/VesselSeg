from ..point3d import Point3D
import numpy as np

def paintBall(IDMap, swc, paint_val):
    rad = swc.rad
    cpos = swc.pos.lst()
    xmin = int(np.floor(max(0, cpos[0] - rad)))
    xmax = int(np.ceil(min(IDMap.shape[0] - 1, cpos[0] + rad)))
    ymin = int(np.floor(max(0, cpos[1] - rad)))
    ymax = int(np.ceil(min(IDMap.shape[1] - 1, cpos[1] + rad)))
    zmin = int(np.floor(max(0, cpos[2] - rad)))
    zmax = int(np.ceil(min(IDMap.shape[2] - 1, cpos[2] + rad)))

    for xi in range(xmin, xmax + 1):
        for yi in range(ymin, ymax + 1):
            for zi in range(zmin, zmax + 1):
                if Point3D(xi, yi, zi).dist(Point3D(cpos)) <= rad:
                    IDMap[xi, yi, zi] = paint_val
    return IDMap
