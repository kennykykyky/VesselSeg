from .point3d import Point3D

class SWCNode(object):
	def __init__(self,cpos,crad,cid=None,ctype=None,cpid=None):
		self.id = cid
		self.type = ctype
		self.pos = cpos
		self.rad = crad
		self.pid = cpid
		#match id to another snake
		self.link_id = None

	@classmethod
	def fromline(cls, line):
		ct = line.split(' ')
		cid = int(ct[0])
		ctype = int(ct[1])
		cpos = Point3D([float(i) for i in ct[2:5]])
		crad = float(ct[5])
		cpid = int(ct[6])
		return cls(cpos,crad,cid,ctype,cpid)

	def getlst(self):
		return [self.id,self.type,self.pos.x,self.pos.y,self.pos.z,self.rad,self.pid]

	def  __repr__(self): 
		reprstr =  'SWCNode: (%.3f, %.3f, %.3f) %.3f'%(self.pos.x,self.pos.y,self.pos.z,self.rad)
		if self.id is not None:
			reprstr += ' ID:%d'%self.id
		if self.type is not None:
			reprstr += ' Type:%d'%self.type
		if self.pid is not None:
			reprstr += ' PID:%d'%self.pid
		if self.link_id is not None:
			reprstr += ' Link to:%d-%d' % (self.link_id[0],self.link_id[1])
		return reprstr
