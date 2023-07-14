import xml.etree.ElementTree as ET
import os
import numpy as np
from .point3d import Point3D
import pdb

class XML:
	def __init__(self,xmlpath=None):
		if xmlpath is not None and os.path.exists(xmlpath):
			self.root = ET.parse(xmlpath).getroot()
			self.xmlpath = xmlpath
		else:
			self.initxml()
			self.xmlpath = xmlpath
		self._cropregion = None
		self._landmark = None

	def initxml(self):
		print('init setting xml')
		self.root = ET.Element('xbel')
		self.root.set("version", "1.0")
		self.root.set("iCafe", "iCafePython 1.0")


	@property
	def cropregion(self):
		if self._cropregion is None:
			self._getcrop()
		return self._cropregion

	@property
	def landmark(self):
		if self._landmark is None:
			self.readLandmark()
		return self._landmark
	@landmark.setter
	def landmark(self, value):
	    self._landmark = value
	    self.update_landmark_node()

	@property
	def res(self):
	    return self.getResolution()

	def _getcrop(self):
		settingroot = self.root
		if settingroot.find('CROP') is None:
			self._cropregion = [0,0,-1,-1] 
		else:
			xmin = int(settingroot.find('CROP').find('item0').text)
			ymin = int(settingroot.find('CROP').find('item1').text)
			xmax = int(settingroot.find('CROP').find('item2').text)
			ymax = int(settingroot.find('CROP').find('item3').text)
			self._cropregion = [xmin,ymin,xmax,ymax]

	def readLandmark(self, IGNOREM3=0):
		settingroot = self.root
		self._landmark = []
		landmarkroot = settingroot.find('Landmark') 
		if landmarkroot is None:
			return 
		else:
			pointnodes = landmarkroot.findall('Point') 
			for nodei in pointnodes:
				ctype = int(nodei.find('type').text)
				clocation = Point3D([float(p) for p in nodei.find('location').text.split(',')])
				if IGNOREM3:
					if ctype in [13,14]:
						continue
				self._landmark.append([ctype,clocation])

	def getLandmark(self,type):
		for li in self.landmark:
			if li[0]==type:
				return li[1]
		return -1

	def getLandmarks(self,type):
		landmark_pos = []
		for li in self.landmark:
			if li[0]==type:
				landmark_pos.append(li[1])
		return landmark_pos

	def update_landmark_node(self):
		if self._landmark is None or len(self._landmark)==0:
			return
		landmarkroot = self.root.find('Landmark') 
		if landmarkroot is not None:
			self.root.remove(landmarkroot)
			#print('clear previous landmarks')
		LandmarkNode = ET.SubElement(self.root, 'Landmark')
		for landi in self._landmark:
			PointNode = ET.SubElement(LandmarkNode, 'Point')
			locationNode = ET.SubElement(PointNode, 'location')
			locationNode.text = ','.join('%.3f'%i for i in landi[1].lst())
			typeNode = ET.SubElement(PointNode, 'type')
			typeNode.text = str(landi[0])

	def readSeqRTM(self,seqname):
		rtm = [[0,0,0] for i in range(4)]
		settingroot = self.root
		SeqRTM = settingroot.find('SeqRTM')
		if SeqRTM is None:
			raise ValueError('No seqRTM defined')
		Seqs = SeqRTM.findall('Seq')
		has_set = False
		for Seq in Seqs:
			Seqname = Seq.find('Seqname')
			if Seqname.text != seqname:
				continue
			for i in range(9):
				rtm[i // 3][i% 3] = float(Seq.find('R' + str(i + 1)).text)
			for i in range(3):
				rtm[3][i%3] = float(Seq.find('T'+str(i+1)).text)
			has_set = True
			break
		if has_set == False:
			raise ValueError('No such Seqname',seqname)
		return rtm

	def removeSeqRTM(self,seqname,SeqRTM=None):
		settingroot = self.root
		if SeqRTM is None:
			SeqRTM = settingroot.find('SeqRTM')
			if SeqRTM is None:
				print('No seq exist')
				return
		extseqs = SeqRTM.findall('Seq')
		for si in extseqs:
			if si.find('Seqname').text == seqname:
				print('remove existing seq',seqname)
				SeqRTM.remove(si)

	def addSeqRTM(self,seqname,rtm):
		settingroot = self.root
		SeqRTM = settingroot.find('SeqRTM') 
		if SeqRTM is None:
			SeqRTM = ET.SubElement(settingroot, 'SeqRTM')
		#check and remove existing seqname
		self.removeSeqRTM(seqname,SeqRTM)

		Seq = ET.SubElement(SeqRTM, 'Seq')
		Seqname = ET.SubElement(Seq, 'Seqname')
		Seqname.text = seqname
		R1 = ET.SubElement(Seq, 'R1')
		R1.text = '%.5f'%rtm[0][0]
		R2 = ET.SubElement(Seq, 'R2')
		R2.text = '%.5f'%rtm[0][1]
		R3 = ET.SubElement(Seq, 'R3')
		R3.text = '%.5f'%rtm[0][2]
		R4 = ET.SubElement(Seq, 'R4')
		R4.text = '%.5f'%rtm[1][0]
		R5 = ET.SubElement(Seq, 'R5')
		R5.text = '%.5f'%rtm[1][1]
		R6 = ET.SubElement(Seq, 'R6')
		R6.text = '%.5f'%rtm[1][2]
		R7 = ET.SubElement(Seq, 'R7')
		R7.text = '%.5f'%rtm[2][0]
		R8 = ET.SubElement(Seq, 'R8')
		R8.text = '%.5f'%rtm[2][1]
		R9 = ET.SubElement(Seq, 'R9')
		R9.text = '%.5f'%rtm[2][2]
		T1 = ET.SubElement(Seq, 'T1')
		T1.text = '%.5f'%rtm[3][0]
		T2 = ET.SubElement(Seq, 'T2')
		T2.text = '%.5f'%rtm[3][1]
		T3 = ET.SubElement(Seq, 'T3')
		T3.text = '%.5f'%rtm[3][2]


	def InitSeqRTM(self, seqname):
		settingroot = self.root
		SeqRTM = settingroot.find('SeqRTM') 
		if SeqRTM is None:
			SeqRTM = ET.SubElement(settingroot, 'SeqRTM')

		extseqs = SeqRTM.findall('Seq')
		for si in extseqs:
			if si.find('Seqname').text == seqname:
				return self.readSeqRTM(seqname)

		rtm = [[1,0,0],
				[0,1,0],
				[0,0,1],
				[0,0,0]]
		self.addSeqRTM(seqname, rtm)
		return rtm


	def writexml(self,path=None):
		self.update_landmark_node()
		if path is None:
			assert self.xmlpath is not None
			path = self.xmlpath
		xml = ET.tostring(self.root)
		with open(path, "w") as myfile:
			myfile.write('<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE xbel>')
			myfile.write(xml.decode("utf-8"))
			myfile.close()
		print('write', path)


	#VW Seg id map
	def loadVWSeg(self):
		settingroot = self.root
		VWSeg = settingroot.find('VWSeg')
		vwsegs = []
		if VWSeg is None:
			return vwsegs
		for Seg in VWSeg.findall('Seg'):
			snakeid = Seg.find('Snakeid').text
			startid = Seg.find('Start').text
			endid = Seg.find('End').text
			sliceid = Seg.find('vwids').text.split(',')[0].split(':')[1]
			qvspath = Seg.find('QVSPath').text
			vwsegs.append((snakeid, startid, endid, sliceid, qvspath))
		return vwsegs

	def removeVWSeg(self):
		settingroot = self.root
		VWSeg = settingroot.find('VWSeg')
		if VWSeg is None:
			return
		ext_segs = VWSeg.findall('Seg')
		for si in ext_segs:
			VWSeg.remove(si)

	def setVWSegs(self,vwsegs):
		# check and remove existing seqname
		self.removeVWSeg()
		for snakeid,startid,endid,sliceid,qvspath in vwsegs:
			self.appendVWSeg(snakeid, startid, endid, sliceid, qvspath)

	def appendVWSeg(self,snakeid,startid,endid,sliceid,qvspath):
		settingroot = self.root
		VWSeg = settingroot.find('VWSeg')
		if VWSeg is None:
			VWSeg = ET.SubElement(settingroot, 'VWSeg')
		Seg = ET.SubElement(VWSeg, 'Seg')
		Start = ET.SubElement(Seg, 'Start')
		Start.text = str(startid)
		End = ET.SubElement(Seg, 'End')
		End.text = str(endid)
		Snakeid = ET.SubElement(Seg, 'Snakeid')
		Snakeid.text = str(snakeid)
		vwids = ET.SubElement(Seg, 'vwids')
		ptids = np.arange(startid,endid)
		sliceids = np.arange(sliceid,sliceid+len(ptids))
		vwids.text = ','.join(['%d:%d'%(ptids[i],sliceids[i]) for i in range(len(ptids))])
		QVSPath = ET.SubElement(Seg, 'QVSPath')
		QVSPath.text = qvspath

	def setResolution(self,res):
		settingroot = self.root
		for resi in settingroot.findall('Resolution'):
			settingroot.remove(resi)
		Resolution = ET.SubElement(settingroot, 'Resolution')
		Res = ET.SubElement(Resolution, 'Res')
		Res.text = '%.5f'%res

	def getResolution(self):
		settingroot = self.root
		Resolution = settingroot.find('Resolution')
		if Resolution is None:
			return None
		return float(Resolution.find('Res').text)
