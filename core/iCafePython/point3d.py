import numpy as np


class Point3D:
	def __init__(self, listinput, pointy=None, pointz=None):
		if type(listinput) in [list, np.ndarray, tuple]:
			self.x = listinput[0]
			self.y = listinput[1]
			self.z = listinput[2]
			self.pos = listinput
		elif type(listinput) in [float, int, np.float64, np.float32] and pointy is not None and pointz is not None:
			self.x = listinput
			self.y = pointy
			self.z = pointz
			self.pos = [listinput, pointy, pointz]
		else:
			print('__init__ unknown type', type(listinput))

	def __repr__(self):
		return 'Point3D(%.3f, %.3f, %.3f)' % (self.x, self.y, self.z)

	def __add__(self, pt2):
		return Point3D([self.x + pt2.x, self.y + pt2.y, self.z + pt2.z])

	def __sub__(self, pt2):
		return Point3D([self.x - pt2.x, self.y - pt2.y, self.z - pt2.z])

	def __mul__(self, scale):
		if type(scale) in [float, int, np.float64, np.float32]:
			return Point3D([self.x * scale, self.y * scale, self.z * scale])
		elif type(scale) in [Point3D, 'iCafe.Point3D']:
			return self.x * scale.x + self.y * scale.y + self.z * scale.z
		else:
			print('__mul__ Unsupport type', type(scale))

	def __truediv__(self, scale):
		if scale == 0:
			print(self.x, self.y, self.z, 'divide Point3D by scale 0')
			scale = 1
		return Point3D([self.x / scale, self.y / scale, self.z / scale])

	def __neg__(self):
		return Point3D([-self.x, -self.y, -self.z])

	def dist(self, pt2):
		return np.linalg.norm((self - pt2).pos)

	def dist2d(self, pt2):
		return np.linalg.norm((self - pt2).pos[:2])

	def path(self, pt2):
		pathlist = []
		nsamples = int(np.floor(self.dist(pt2)))
		norm_vec = (pt2 - self) / nsamples
		for stepi in range(nsamples):
			pathlist.append((self + norm_vec * stepi))
		pathlist.append(pt2)
		return pathlist

	def vecLenth(self):
		return np.linalg.norm(self.pos)

	def outOfBound(self, imgshape):
		if self.x < 0 or self.x > imgshape[0] - 1:
			return 1
		if self.y < 0 or self.y > imgshape[1] - 1:
			return 1
		if self.z < 0 or self.z > imgshape[2] - 1:
			return 1
		return 0

	def outOfBox(self, boxshape, allowance=0):
		#box shape [xmin,xmax,ymin,ymax,zmin,zmax]
		if self.x < boxshape[0] - allowance or self.x > boxshape[1] - 1 + allowance:
			return 1
		if self.y < boxshape[2] - allowance or self.y > boxshape[3] - 1 + allowance:
			return 1
		if self.z < boxshape[4] - allowance or self.z > boxshape[5] - 1 + allowance:
			return 1
		return 0

	def norm(self):
		abnorm = np.linalg.norm(self.pos)
		if abnorm == 0:
			# print('vec norm magnitude 0')
			return Point3D(self.pos)
		return Point3D(self.pos / abnorm)

	def lst(self):
		return [self.x, self.y, self.z]

	def intlst(self):
		return [int(round(self.x)), int(round(self.y)), int(round(self.z))]

	def toIntPos(self):
		self.x = int(round(self.x))
		self.y = int(round(self.y))
		self.z = int(round(self.z))

	def prod(self, pt2):
		return self.x * pt2.x + self.y * pt2.y + self.z * pt2.z

	def getAngle(self, pt2):
		#in rad
		pt2 = pt2.norm()
		self = self.norm()
		return np.arccos(self.prod(pt2))

	def getAngleDeg(self, pt2):
		#in degree
		return self.getAngle(pt2) / np.pi * 180

	def hashPos(self):
		return '-'.join(['%.3f' % i for i in self.lst()])

	def intHashPos(self):
		return '-'.join(['%d' % i for i in self.lst()])

	def posMatch(self, points):
		bestmatchscore = 0
		bestid = -1
		for idx, pi in enumerate(points):
			cmatchscore = self * pi
			if cmatchscore > bestmatchscore:
				bestmatchscore = cmatchscore
				bestid = idx
		return bestid

	#restrict pos within limits
	def bound(self, upperx, uppery, upperz, lowerx=0, lowery=0, lowerz=0):
		self.x = max(self.x, lowerx)
		self.y = max(self.y, lowery)
		self.z = max(self.z, lowerz)
		self.x = min(self.x, upperx - 1)
		self.y = min(self.y, uppery - 1)
		self.z = min(self.z, upperz - 1)

	def boundList(self, shape):
		if len(shape) < 3:
			raise ValueError('no enough shape dimension')
		self.bound(shape[0], shape[1], shape[2])

	def findRad(self, simg):
		pti = self.lst()
		MAX_RAD = 10
		for rad in range(1, MAX_RAD):
			ct = 6
			for ofi in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]:
				ptx = int(round(pti[0] + rad * ofi[0]))
				pty = int(round(pti[1] + rad * ofi[1]))
				ptz = int(round(pti[2] + rad * ofi[2]))
				if Point3D(ptx, pty, ptz).outOfBound(simg.shape):
					ct -= 1
				elif simg[ptx, pty, ptz] != True:
					ct -= 1
			if ct < 3:
				return rad + ct / 6
		return rad

	def hasNaN(self):
		if np.isnan(self.x) or np.isnan(self.y) or np.isnan(self.z):
			return True
		else:
			return False

	def copy(self):
		return Point3D(self.x, self.y, self.z)

	def __init__(self, listinput, pointy=None, pointz=None):
		if type(listinput) in [list,np.ndarray,tuple]:
			self.x = listinput[0]
			self.y = listinput[1]
			self.z = listinput[2]
			self.pos = listinput
		elif type(listinput) in [float,int,np.float64,np.float32] and pointy is not None and pointz is not None:
			self.x = listinput
			self.y = pointy
			self.z = pointz
			self.pos = [listinput,pointy,pointz]
		else:
			print('__init__ unknown type',type(listinput))
	def  __repr__(self):
		return 'Point3D(%.3f, %.3f, %.3f)'%(self.x,self.y,self.z)
	def __add__(self,pt2):
		return Point3D([self.x+pt2.x,self.y+pt2.y,self.z+pt2.z])
	def __sub__(self,pt2):
		return Point3D([self.x-pt2.x,self.y-pt2.y,self.z-pt2.z])
	def __mul__(self,scale):
		if type(scale) in [float,int,np.float64,np.float32]:
			return Point3D([self.x*scale,self.y*scale,self.z*scale])
		elif type(scale) in [Point3D,'iCafe.Point3D']:
			return self.x*scale.x+self.y*scale.y+self.z*scale.z
		else:
			print('__mul__ Unsupport type',type(scale))
	def __truediv__(self,scale):
		if scale ==0:
			print(self.x,self.y,self.z,'divide Point3D by scale 0')
			scale = 1
		return Point3D([self.x/scale,self.y/scale,self.z/scale])
	def __neg__(self):
		return Point3D([-self.x,-self.y,-self.z])
	def dist(self,pt2):
		return np.linalg.norm((self-pt2).pos)
	def dist2d(self,pt2):
		return np.linalg.norm((self-pt2).pos[:2])
	def path(self,pt2):
		pathlist = []
		nsamples = int(np.floor(self.dist(pt2)))
		norm_vec = (pt2-self)/nsamples
		for stepi in range(nsamples):
			pathlist.append((self+norm_vec*stepi))
		pathlist.append(pt2)
		return pathlist
	def vecLenth(self):
		return np.linalg.norm(self.pos)
	def outOfBound(self,imgshape):
		if self.x<0 or self.x>imgshape[0]-1:
			return 1
		if self.y<0 or self.y>imgshape[1]-1:
			return 1
		if self.z<0 or self.z>imgshape[2]-1:
			return 1
		return 0
	def outOfBox(self,boxshape,allowance=0):
		#box shape [xmin,xmax,ymin,ymax,zmin,zmax]
		if self.x<boxshape[0]-allowance or self.x>boxshape[1]-1+allowance:
			return 1
		if self.y<boxshape[2]-allowance or self.y>boxshape[3]-1+allowance:
			return 1
		if self.z<boxshape[4]-allowance or self.z>boxshape[5]-1+allowance:
			return 1
		return 0
	def norm(self):
		abnorm = np.linalg.norm(self.pos)
		if abnorm == 0:
			# print('vec norm magnitude 0')
			return Point3D(self.pos)
		return Point3D(self.pos/abnorm)
	def lst(self):
		return [self.x,self.y,self.z]
	def intlst(self):
		return [int(round(self.x)),int(round(self.y)),int(round(self.z))]
	def toIntPos(self):
		self.x = int(round(self.x))
		self.y = int(round(self.y))
		self.z = int(round(self.z))
	def prod(self,pt2):
		return self.x*pt2.x+self.y*pt2.y+self.z*pt2.z
	def getAngle(self,pt2):
		#in rad
		pt2 = pt2.norm()
		self = self.norm()
		return np.arccos(self.prod(pt2))
	def getAngleDeg(self,pt2):
		#in degree
		return self.getAngle(pt2)/np.pi*180
	def hashPos(self):
		return '-'.join(['%.3f'%i for i in self.lst()])
	def intHashPos(self):
		return '-'.join(['%d'%i for i in self.lst()])
	def posMatch(self,points):
		bestmatchscore = 0
		bestid = -1
		for idx,pi in enumerate(points):
			cmatchscore = self*pi
			if cmatchscore>bestmatchscore:
				bestmatchscore = cmatchscore 
				bestid = idx
		return bestid
	#restrict pos within limits
	def bound(self,upperx,uppery,upperz,lowerx=0,lowery=0,lowerz=0):
		self.x = max(self.x, lowerx)
		self.y = max(self.y, lowery)
		self.z = max(self.z, lowerz)
		self.x = min(self.x, upperx-1)
		self.y = min(self.y, uppery-1)
		self.z = min(self.z, upperz-1)

	def boundList(self,shape):
		if len(shape)<3:
			raise ValueError('no enough shape dimension')
		self.bound(shape[0],shape[1],shape[2])

	def findRad(self,simg):
		pti = self.lst()
		MAX_RAD = 10
		for rad in range(1,MAX_RAD):
			ct = 6
			for ofi in [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]:
				ptx = int(round(pti[0]+rad*ofi[0]))
				pty = int(round(pti[1]+rad*ofi[1]))
				ptz = int(round(pti[2]+rad*ofi[2]))
				if Point3D(ptx,pty,ptz).outOfBound(simg.shape):
					ct -= 1
				elif simg[ptx,pty,ptz] != True:
					ct -= 1
			if ct<3:
				return rad+ct/6
		return rad

	def hasNaN(self):
		if np.isnan(self.x) or np.isnan(self.y) or np.isnan(self.z):
			return True
		else:
			return False

	def copy(self):
		return Point3D(self.x,self.y,self.z)
