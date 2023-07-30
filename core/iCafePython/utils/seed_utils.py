import os
from ..point3d import Point3D

def loadSeedsFile(seedfilename):
    seeds = []
    if not os.path.exists(seedfilename):
        print('seed file not exist',seedfilename)
        return seeds
    with open(seedfilename,'r') as fp:
        for line in fp:
            items = line.split(' ')
            if len(items)!=3:
                print('err seed line',line)
                continue
            seeds.append(Point3D(float(items[0]),float(items[1]),float(items[2])))
    return seeds

def writeSeedsFile(path,seeds):
	if seeds is None:
		print('seeds not loaded')
		return
	with open(path,'w') as fp:
		for seedi in seeds:
			fp.write('%.3f %.3f %.3f\n'%tuple(seedi.lst()))
