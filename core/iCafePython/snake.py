import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .swcnode import SWCNode
from .point3d import Point3D
# from sklearn.linear_model import HuberRegressor
from .utils.img_utils import paint_dist_unique

class Snake:
	def __init__(self,swcnodelist=None,type = 0):
		if swcnodelist is None:
			self.snake = []
		else:
			self.snake = copy.copy(swcnodelist)
		self.type = type
		self.reset()
		self.id = None

	def  __repr__(self): 
		return 'Snake with %d points, type %d'%(len(self.snake),self.type)

	@classmethod
	def fromList(cls,snake_list):
		swc_list = []
		for pti in snake_list:
			swc_list.append(SWCNode(Point3D(pti),1))
		return cls(swc_list)

	def reset(self):
		self._NP = None
		self._volume = None
		# tortuosity
		self._tot = None
		# average rad for all points
		self._arad = None
		# bounding box around all pts
		self._box = None

	@property
	def NP(self):
		self._NP = len(self.snake)
		return self._NP

	@property
	def length(self):
		return self._getLength()

	@property
	def volume(self):
		if self._volume is None:
			self._volume = self._getVolume()
		return self._volume

	@property
	def tot(self):
		if self._tot is None:
			self._tot = self._getTot()
		return self._tot

	@property
	def arad(self):
		if self._arad is None:
			self._arad = self._getRad()
		return self._arad

	@property
	def box(self):
		if self._box is None:
			xmin = np.inf
			ymin = np.inf
			zmin = np.inf
			xmax = 0
			ymax = 0
			zmax = 0
			for i in range(self.NP):
				if self.snake[i].pos.x < xmin:
					xmin = self.snake[i].pos.x
				if self.snake[i].pos.x > xmax:
					xmax = self.snake[i].pos.x
				if self.snake[i].pos.y < ymin:
					ymin = self.snake[i].pos.y
				if self.snake[i].pos.y > ymax:
					ymax = self.snake[i].pos.y
				if self.snake[i].pos.z < zmin:
					zmin = self.snake[i].pos.z
				if self.snake[i].pos.z > zmax:
					zmax = self.snake[i].pos.z
			self._box = [xmin,xmax,ymin,ymax,zmin,zmax]
		return self._box

	@property
	def box_vol(self):
		box = self.box
		return (box[1]-box[0])*(box[3]-box[2])*(box[5]-box[4])

	@property
	def link_pts(self):
		_linked_pts = 0
		for i in range(self.NP):
			if self.snake[i].link_id is not None:
				_linked_pts += 1
		return _linked_pts

	@property
	def link_dist(self):
		dists = []
		for i in range(self.NP):
			if self.snake[i].link_id is not None:
				dists.append(self.snake[i].link_id[2])
			else:
				dists.append(np.nan)
		return dists

	@property
	def ct(self):
		pos = np.array([self.snake[i].pos.lst() for i in range(self.NP)])
		return Point3D(np.mean(pos,axis=0))

	def __len__(self):
		return self.NP

	def __getitem__(self, key):
		return self.snake[key]

	def copy(self):
		return copy.deepcopy(self)

	def add(self,swcnode):
		self.snake.append(swcnode)

	def append(self,swcnode):
		self.snake.append(swcnode)

	def insert(self,i,elem):
		self.snake.insert(i,elem)

	def addSWC(self,pos,rad=1,cid=None,ctype=None,cpid=None):
		self.snake.append(SWCNode(pos,rad,cid,ctype,cpid))

	def addPt(self,x,y,z,rad=1,cid=None,ctype=None,cpid=None):
		self.snake.append(SWCNode(Point3D(x,y,z),rad,cid,ctype,cpid))

	def dist(self,snakej):
		mindist = np.inf
		for pti in range(self.NP):
			for ptj in range(snakej.NP):
				cdist = self.snake[pti].pos.dist(snakej[ptj].pos)
				if cdist<mindist:
					mindist = cdist
		return mindist

	def mergeSnake(self,snakej,reverse=False,append=True):
		if reverse:
			for ptj in range(snakej.NP-1,-1,-1):
				if append:
					self.snake.append(snakej[ptj])
				else:
					self.snake.insert(0,snakej[ptj])
		else:
			for ptj in range(snakej.NP):
				if append:
					self.snake.append(snakej[ptj])
				else:
					self.snake.insert(0,snakej[ptj])
		#next time use box, will recalculate its space
		self.reset()

	def mergeSnakeA(self,snakej):
		dist1 = self.snake[0].pos.dist(snakej.snake[0].pos)
		dist2 = self.snake[0].pos.dist(snakej.snake[-1].pos)
		dist3 = self.snake[-1].pos.dist(snakej.snake[0].pos)
		dist4 = self.snake[-1].pos.dist(snakej.snake[-1].pos)
		mind = min(dist1,dist2,dist3,dist4)
		if mind == dist1:
			self.mergeSnake(snakej, reverse=False, append=False)
		elif mind == dist2:
			self.mergeSnake(snakej, reverse=True, append=False)
		elif mind == dist3:
			self.mergeSnake(snakej, reverse=False, append=True)
		elif mind == dist4:
			self.mergeSnake(snakej, reverse=True, append=True)

	def trimSnake(self,ptj,reverse = False,copy=False):
		assert ptj>=0 and ptj<self.NP
		if copy:
			csnake =  self.copy()
			csnake.trimSnake(ptj,reverse)
			csnake.reset()
			return csnake
		else:
			if reverse == False:
				self.snake = self.snake[:ptj]
			else:
				self.snake = self.snake[ptj:]
			self.reset()


	def branchSnake(self, target_swc, pti = 0):
		ori_end = self.snake[pti].pos
		match_dist = target_swc.pos.dist(ori_end)
		if pti==0:
			if match_dist < 1:
				# change pt
				self.snake[0].pos = target_swc.pos
			else:
				# add pt
				self.snake.insert(0, target_swc)
		else:
			if match_dist < 1:
				# change pt
				self.snake[-1].pos = target_swc.pos
			else:
				# add pt
				self.snake.append(target_swc)
		self.reset()

	def removeDuplicatePts(self):
		del_pts = []
		for pti in range(1,self.NP):
			ppos = self.snake[pti-1].pos
			cpos = self.snake[pti].pos
			if cpos.dist(ppos)<0.001:
				print('del deuplicate pti',pti)
				del_pts.insert(0,pti)
		for pti in del_pts:
			del self.snake[pti]

	def removeSelfLoop(self):
		del_ptis = []
		pos_dict = {}
		message = 'loop del'
		for pti in range(self.NP):
			cpos = self.snake[pti].pos
			hash_pos = cpos.hashPos()
			if hash_pos not in pos_dict:
				pos_dict[hash_pos] = pti
			else:
				lpos_id = pos_dict[hash_pos]
				del_ptis.extend(np.arange(lpos_id,pti).tolist())
				message = message+' %d-%d,'%(lpos_id,pti)
		self.reset()
		if len(del_ptis)==0:
			return self
		else:
			valid_snake = Snake()
			for pti in range(self.NP):
				if pti not in del_ptis:
					valid_snake.add(self.snake[pti])
			self.snake = valid_snake.snake
			print(message)
			return valid_snake

	#part of snake overlap with previous points
	def removeSelfOverlap(self):
		traced_pts = {}
		acc_len = self.getAccLenArray()
		break_pts = [0]
		for pti in range(self.NP):
			int_pos = self.snake[pti].pos.intHashPos()
			cpos = self.snake[pti].pos.intlst()
			crad = int(np.floor(self.snake[pti].rad))
			if int_pos not in traced_pts:
				traced_pts[int_pos] = pti
				#paint ball
				for xi in range(-crad,crad+1):
					for yi in range(-crad,crad+1):
						for zi in range(-crad,crad+1):
							target_pos = Point3D(cpos[0]+xi,cpos[1]+yi,cpos[2]+zi)
							if target_pos.dist(self.snake[pti].pos)>crad:
								continue
							if target_pos.intHashPos() not in traced_pts:
								traced_pts[cpos[0]+xi,cpos[1]+yi,cpos[2]+zi] = pti
			else:
				if acc_len[pti] - acc_len[traced_pts[int_pos]]>crad and pti-traced_pts[int_pos]>5:
					print('pt',pti,'hit previous pt',traced_pts[int_pos])
					break_pts.append((pti+traced_pts[int_pos])//2)
					traced_pts = {}
		break_pts.append(self.NP)
		return break_pts


	def trimRange(self,box):
		valid_snake = Snake()
		for pti in range(self.NP):
			if self.snake[pti].pos.outOfBox(box):
				print(self.snake[pti].pos,'out of box')
				continue
			valid_snake.add(self.snake[pti])
		return valid_snake

	def getAccLen(self, ptid=-1):
		#accumulated length
		if ptid == -1:
			ptid = self.NP-1
		if ptid >= self.NP:
			print('pt id out of bound')
			return -1
		acclen = 0
		for i in range(1,ptid+1):
			acclen += self.snake[i].pos.dist(self.snake[i-1].pos)
		return acclen

	def getAccLenArray(self):
		acclen = np.zeros((self.NP))
		for i in range(1,self.NP):
			acclen[i] = acclen[i-1] + self.snake[i].pos.dist(self.snake[i-1].pos)
		return acclen

	def _getLength(self):
		return self.getAccLen(-1)

	def _getVolume(self):
		accvol = 0
		if self.NP<2:
			return accvol
		for i in range(1,self.NP):
			clen = self.snake[i].pos.dist(self.snake[i-1].pos)
			carea = math.pi*(self.snake[i].rad)**2
			parea = math.pi*(self.snake[i-1].rad)**2
			accvol += clen*(carea+parea)/2
		return accvol

	def _getTot(self):
		if self.NP<1:
			return 0
		htlength = self.snake[0].pos.dist(self.snake[self.NP-1].pos)
		return self.length/htlength
		self._tot = 0

	def _getRad(self):
		if self.NP<1:
			return 0
		rads = [snakei.rad for snakei in self.snake]
		return np.mean(rads)

	def getNorm(self, ptid):
		#get norm direction from given point id on the snake
		csnakepts = self.NP
		if ptid>=csnakepts:
			print('ptid over NP')
			return
		if ptid == 0:
			if len(self.snake) == 1: # Kaiyu add
				#raise ValueError('snake len 1')
				normdirect = self.snake[0].pos
			else:
				normdirect = self.snake[1].pos - self.snake[0].pos
		elif ptid == csnakepts-1:
			normdirect = self.snake[csnakepts-1].pos - self.snake[csnakepts-2].pos
		else:
			normdirect = self.snake[ptid+1].pos - self.snake[ptid].pos
		return normdirect

	@property
	def xlist(self):
		return [self.snake[i].pos.x for i in range(self.NP)]

	@property
	def ylist(self):
		return [self.snake[i].pos.y for i in range(self.NP)]

	@property
	def zlist(self):
		return [self.snake[i].pos.z for i in range(self.NP)]

	@property
	def radList(self):
		return [self.snake[i].rad for i in range(self.NP)]

	def plot(self):
		plt.plot(self.xlist, [self.getAccLen(i) for i in range(self.NP)], label='x')
		plt.plot(self.ylist, [self.getAccLen(i) for i in range(self.NP)], label='y')
		plt.plot(self.zlist, [self.getAccLen(i) for i in range(self.NP)], label='z')
		plt.legend(loc='lower right')
		plt.gca().invert_yaxis()
		plt.show()

	def resampleSnake(self,gap = None):
		if self.NP<2:
			return self
		ressnake = []
		d = self.getAccLenArray()
		if gap is None:
			samples = len(d)
		else:
			samples = int(d[-1]//gap)
		#at least three points after resample
		samples = max(3,samples)
		x = []
		y = []
		z = []
		r = []
		#fill xyzr
		for i in range(self.NP):
			x.append(self.snake[i].pos.x)
			y.append(self.snake[i].pos.y)
			z.append(self.snake[i].pos.z)
			r.append(self.snake[i].rad)
		
		fx = interp1d(d,x)
		fy = interp1d(d,y)
		fz = interp1d(d,z)
		fr = interp1d(d,r)
		dsample = np.linspace(0, d[-1], samples)
		xi = fx(dsample)
		yi = fy(dsample)
		zi = fz(dsample)
		ri = fr(dsample)
		for i in range(len(dsample)):
			ressnake.append(SWCNode(Point3D(xi[i],yi[i],zi[i]),ri[i]))
		#remain head and tail swc type
		ressnake[0].type = self.snake[0].type
		ressnake[-1].type = self.snake[-1].type
		return Snake(ressnake)

	#resample based on z direction, asuuming no duplicate z
	def resampleSnakeZunit(self):
		ressnake = []
		snake_z = [zi.pos.z for zi in self.snake]

		# at least three points after resample
		x = []
		y = []
		z = []
		r = []
		# fill xyzr
		for i in range(self.NP):
			x.append(self.snake[i].pos.x)
			y.append(self.snake[i].pos.y)
			z.append(self.snake[i].pos.z)
			r.append(self.snake[i].rad)

		fx = interp1d(snake_z, x)
		fy = interp1d(snake_z, y)
		fz = interp1d(snake_z, z)
		fr = interp1d(snake_z, r)
		zmin = int(np.ceil(np.min(snake_z)))
		zmax = int(np.floor(np.max(snake_z)))
		dsample = np.linspace(zmin, zmax, zmax-zmin+1)
		xi = fx(dsample)
		yi = fy(dsample)
		zi = fz(dsample)
		ri = fr(dsample)
		for i in range(len(dsample)):
			ressnake.append(SWCNode(Point3D(xi[i], yi[i], zi[i]), ri[i]))
		return Snake(ressnake)


	#resample based on norm direction, asuuming no duplicate points on norm direction
	def resampleSnakeNormUnit(self,kind='linear'):
		norm = self.snake[-1].pos - self.snake[0].pos
		norm_unit = norm.norm()
		ressnake = []
		snake_norm = [(zi.pos-self.snake[0].pos).prod(norm_unit) for zi in self.snake]
		# at least three points after resample
		x = []
		y = []
		z = []
		r = []
		# fill xyzr
		for i in range(self.NP):
			x.append(self.snake[i].pos.x)
			y.append(self.snake[i].pos.y)
			z.append(self.snake[i].pos.z)
			r.append(self.snake[i].rad)
		if kind=='huber':
			fx = HuberRegressor().fit(np.array(snake_norm).reshape(-1, 1), x)
			fy = HuberRegressor().fit(np.array(snake_norm).reshape(-1, 1), y)
			fz = HuberRegressor().fit(np.array(snake_norm).reshape(-1, 1), z)
			fr = HuberRegressor().fit(np.array(snake_norm).reshape(-1, 1), r)
		else:
			fx = interp1d(snake_norm, x, kind=kind)
			fy = interp1d(snake_norm, y, kind=kind)
			fz = interp1d(snake_norm, z, kind=kind)
			fr = interp1d(snake_norm, r, kind=kind)
		norm_min = int(np.ceil(np.min(snake_norm)))
		norm_max = int(np.floor(np.max(snake_norm)))
		#dsample = np.linspace(norm_min, norm_max, norm_max - norm_min + 1)
		dsample = np.linspace(norm_min, norm_max, len(snake_norm))
		if kind=='huber':
			xi = fx.predict(dsample.reshape(-1, 1))
			yi = fy.predict(dsample.reshape(-1, 1))
			zi = fz.predict(dsample.reshape(-1, 1))
			ri = fr.predict(dsample.reshape(-1, 1))
		else:
			xi = fx(dsample)
			yi = fy(dsample)
			zi = fz(dsample)
			ri = fr(dsample)
		for i in range(len(dsample)):
			ressnake.append(SWCNode(Point3D(xi[i], yi[i], zi[i]), ri[i]))
		return Snake(ressnake)

	# by segments
	def resampleSnakeNormUnitSeg(self,seg=100,kind='linear'):
		if self.NP<=seg:
			return self.resampleSnakeNormUnit()
		else:
			resampled_snake = Snake()
			for segi in range(self.NP//seg):
				seg_s = segi * seg
				seg_e = (segi+1) * seg
				cur_snake = self.subSnake(seg_s,seg_e)
				resampled_cur_snake = cur_snake.resampleSnakeNormUnit(kind=kind)
				resampled_snake.mergeSnake(resampled_cur_snake)
			#last seg
			if self.NP%seg!=0:
				seg_s = self.NP - seg
				seg_e = self.NP
				cur_snake = self.subSnake(seg_s, seg_e)
				resampled_cur_snake = cur_snake.resampleSnakeNormUnit(kind=kind)
				resampled_cur_snake.trimSnake(self.NP//seg*seg-seg_s,reverse=True)
				resampled_snake.mergeSnake(resampled_cur_snake)
		return resampled_snake

	def reverseSnake(self):
		self.snake = self.snake[::-1]
		return self

	def subSnake(self, startid, endid=-1):
		if endid == -1:
			endid = self.NP
		return Snake(self.snake[startid:endid])

	def inSnake(self,pos,thres=3):
		for pti in range(self.NP):
			if pos.dist(self.snake[pti].pos)<thres:
				return True
		return False

	def arrangeSnakeDirection(self):
		if self.snake[0].rad < self.snake[-1].rad:
			self.reverseSnake()

	def movingAvgSnake(self,movingavg=5):
		snakema = []
		for nodeidx in range(self.NP):
			if nodeidx<=movingavg//2 or nodeidx>=self.NP-movingavg//2:
				snakema.append(copy.copy(self.snake[nodeidx]))
			else:
				avgx = np.mean([self.snake[idx].pos.x for idx in range(nodeidx-movingavg//2,nodeidx+movingavg//2+1)])
				avgy = np.mean([self.snake[idx].pos.y for idx in range(nodeidx-movingavg//2,nodeidx+movingavg//2+1)])
				avgz = np.mean([self.snake[idx].pos.z for idx in range(nodeidx-movingavg//2,nodeidx+movingavg//2+1)])
				avgr = np.mean([self.snake[idx].rad for idx in range(nodeidx-movingavg//2,nodeidx+movingavg//2+1)])
				cnode = SWCNode(Point3D(avgx,avgy,avgz),avgr)
				snakema.append(cnode)
		return Snake(snakema)

	def trimHeadRad(self):
		snake = copy.copy(self.snake)
		if self.NP<8:
			return snake
		if snake[0].rad>snake[3].rad*1.5:
			snake[0].rad = snake[3].rad
			snake[1].rad = snake[3].rad
			snake[2].rad = snake[3].rad
		if snake[-1].rad>snake[-4].rad*1.5:
			snake[-1].rad = snake[-4].rad
			snake[-2].rad = snake[-4].rad
			snake[-3].rad = snake[-4].rad
		return Snake(snake)

	#refresh radius based on binary map
	def fitRad(self,simg):
		for pti in range(self.NP):
			crad = self.snake[pti].pos.findRad(simg)
			self.snake[pti].rad = crad

	def nearestPt(self,pos,thres=1):
		idx = -1
		min_dist = np.inf
		for pti in range(self.NP):
			cdist = self.snake[pti].pos.dist(pos)
			if cdist>thres:
				continue
			if cdist<min_dist:
				min_dist = cdist
				idx = pti
		return idx

	# find branch point swc to the target snake
	def findBranchPt(self, pos):
		idx = self.nearestPt(pos,np.inf)
		return self.snake[idx]

	def posLoss(self,b=5):
		def sumSSpos(pos,b):
			pos = np.array(pos)
			sx = pos[1:] - pos[:-1]
			ssx = sx[1:] - sx[:-1]
			return np.sum(abs(sx)) + b * np.sum(abs(ssx))
		posx = self.xlist
		posy = self.ylist
		posz = self.zlist
		E_int = sumSSpos(posx,b) + sumSSpos(posy,b) + sumSSpos(posz,b)
		return E_int

	def matchComp(self, ref_snakelist):
		#tracking match
		TP = 0
		FN = 0
		T = self.NP
		#ID match
		IDTP = 0
		IDFN = 0
		IDFP = 0
		self.resampleSnake(1)
		#set of matched points from ref snakelist, if pt in test snakelist has no match, it is FP
		matched_ids = set()
		#all snakei within radius of gt
		match_snake_ids = [None for i in range(self.NP)]
		#final assignment of snakei to each pti
		snakeid_assigned = [None] * self.NP
		for pti in range(self.NP):
			cpos = self.snake[pti].pos
			crad = self.snake[pti].rad
			#+1 to allow more space in box thres
			match_snake_ids[pti] = ref_snakelist.matchPts(cpos,crad+1)
			if len(match_snake_ids[pti])>0:
				TP += 1
				for match_id in match_snake_ids[pti]:
					# 0: snakeid, 1: ptid, 2: dist
					matched_ids.add(str(match_id[0]) + '-' + str(match_id[1]))
			else:
				FN += 1
				snakeid_assigned[pti] = -1

		#assign match by counting most match snakeid
		rep = 0
		while None in snakeid_assigned:
			snake_candidates = {}
			for pti in range(self.NP):
				if snakeid_assigned[pti] is not None:
					continue
				for match_snake_id in match_snake_ids[pti]:
					ref_snakei = match_snake_id[0]
					if ref_snakei not in snake_candidates:
						snake_candidates[ref_snakei] = 1
					else:
						snake_candidates[ref_snakei] += 1
			#print('snakeid',snake_candidates)
			frequent_snake = max(snake_candidates, key=lambda k: snake_candidates[k])
			#assign
			for pti in range(self.NP):
				for match_snake_id in match_snake_ids[pti]:
					ref_snakei = match_snake_id[0]
					ref_pti = match_snake_id[1]
					if ref_snakei == frequent_snake:
						snakeid_assigned[pti] = frequent_snake
						#if points on the test snakelist has no link_id, it is IDFP
						ref_snakelist[ref_snakei][ref_pti].link_id = (self.id,pti,match_snake_id[2])
						self.snake[pti].link_id = match_snake_id
						#print('lk',ref_snakei,ref_pti)
			rep += 1
		#print('snakeid_assigned final',snakeid_assigned)

		if TP/T>0.8:
			MT = 1
		else:
			MT = 0
		if FN/T>0.8:
			ML = 1
		else:
			ML = 0

		fragmentation = 0
		IDS = 0
		prev_snakej = snakeid_assigned[0]
		for i in range(1,len(snakeid_assigned)):
			if prev_snakej==-1 and snakeid_assigned[i]!=-1 or prev_snakej!=-1 and snakeid_assigned[i]==-1:
				fragmentation += 1
			if prev_snakej!=snakeid_assigned[i]:
				#print('ids',prev_snakej,snakeid_assigned[i])
				IDS += 1
			prev_snakej = snakeid_assigned[i]

		#count id
		for pti in range(len(snakeid_assigned)):
			if snakeid_assigned[pti] == -1:
				IDFN += 1
			else:
				IDTP += 1
		return TP,FN,matched_ids,MT,ML,fragmentation, IDS, IDTP, IDFN

	def checkSharpTurn(self,angle_th):
		point_dist = 2
		sharp_turn = False
		angle = 0
		for i in range(point_dist,self.NP-point_dist):
			V1 = self.snake[i+point_dist].pos - self.snake[i].pos
			V2 = self.snake[i].pos - self.snake[i-point_dist].pos
			angle_temp = V1.getAngle(V2)/np.pi*180
			if angle_temp > angle:
				angle = angle_temp
		if angle > angle_th:
			print('sharp turn',angle)
			sharp_turn = True
		return sharp_turn

	def checkNaN(self):
		for i in range(self.NP):
			if self.snake[i].pos.hasNaN():
				return True
		return False

	def labelImgEncoding(self,label_img,paintVal,coding_radius=-1):
		if 0 < coding_radius < 1:
			coding_radius_scale = coding_radius
			coding_radius = -1
		else:
			coding_radius_scale = 1
		resamp = self.copy()
		if coding_radius!=-1 and self.length/self.NP>coding_radius:
			resamp = resamp.resampleSnake(coding_radius)
		for pti in range(resamp.NP):
			pos = resamp.snake[pti].pos
			if coding_radius ==-1:
				coding_radius = max(1,resamp.snake[pti].rad*coding_radius_scale)
			# paint within radius at pos position
			paint_dist_unique(label_img, pos, coding_radius, paintVal,markduplicate=False)