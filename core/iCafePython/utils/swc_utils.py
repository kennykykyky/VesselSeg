from ..definition import matchvestype
from ..swcnode import SWCNode
from ..snake import Snake
from ..snakelist import SnakeList
import copy
import os

def loadSWCFile(swcfilename):
	#node info in SWCNode, in and output. Load swclist
	swclist = []
	#snakelist representation, link with swclist
	snakelist = SnakeList()
	if not os.path.exists(swcfilename):
		print('not exist', swcfilename)
		return snakelist, swclist
	with open(swcfilename,'r') as fp:
		for line in fp:
			swclist.append(SWCNode.fromline(line))
	temp_Snake = []
	start = swclist[0].type # Kaiyu Added
	for i in range(len(swclist)):
		if i > 0 and swclist[i].pid == -1:
			#end last snake
			new_Snake = Snake(temp_Snake) # Kaiyu Added to read the vessel type info from begin and end swc node
			new_Snake.type = matchvestype(start, swclist[i-1].type) # Kaiyu Added
			start = swclist[i].type # Kaiyu Added
			snakelist.addSnake(new_Snake)
			temp_Snake.clear()
			#add new point
			temp_Snake.append(swclist[i])
		else:
			temp_Snake.append(swclist[i])
		#end of the swc file
		if i == len(swclist) - 1:
			snakelist.addSnake(Snake(temp_Snake))
			temp_Snake.clear()
	return snakelist, swclist


def writeSWCFile(path,snakelist):
	cid = 1
	cpid = -1
	with open(path,'w') as fp:
		for snakei in snakelist:
			if len(snakei)<3:
				print('skip snake with nodes less than 3')
				continue

			for nodi in range(len(snakei)):
				swcnodei = snakei[nodi]
				swcnodei.id = cid
				cid += 1
				swcnodei.pid = cpid
				if swcnodei.type is None:
					swcnodei.type = 0
				fp.write('%d %d %.3f %.3f %.3f %.3f %d\n'%tuple(swcnodei.getlst()))
				cpid=swcnodei.id
			cpid = -1
	print('swc saved',path)

def getUniqueSWCFromPtlist(ptlist,swcnodeori):
	swcnode = copy.copy(swcnodeori)
	for swcid in range(len(ptlist)):
		if ptlist[swcid].pos.dist(swcnode.pos)==0:
			#print(swcnode.pos,'id change to ',ptlist[swcid].id)
			swcnode.id = ptlist[swcid].id
			return swcnode
	return swcnode

