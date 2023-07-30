import os
from .point3d import Point3D
from .utils.seed_utils import loadSeedsFile, writeSeedsFile

def loadSeeds(self):
	seedfilename = os.path.join(self.path, 'seed_TH_' + self.filename_solo + '.txt')
	self._seeds = loadSeedsFile(seedfilename)

def addSeed(self,seedpoint3d):
	self._seeds.append(seedpoint3d)

def setSeeds(self,newseedlist):
	self._seeds = newseedlist

def clearSeeds(self):
	self._seeds = []

def setSeedsSnakeList(self,snakelist):
	self.clearSeeds()
	for snakeid in range(snakelist.NSnakes):
		for pti in range(snakelist[snakeid].NP):
			self.addSeed(snakelist[snakeid][pti].pos)

def writeSeeds(self,path=None):
	if self._seeds is None:
		print('seeds not loaded')
		return
	if path is None:
		path = os.path.join(self.path,'seed_TH_' + self.filename_solo + '.txt')
	writeSeedsFile(path,self._seeds)
