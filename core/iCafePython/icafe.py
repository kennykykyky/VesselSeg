import os
import shutil
import pydicom
from .xmlclass import XML
from .cascade.cas import CASCADE
from .reg.reg import Reg
from .definition import BOITYPENUM,VESTYPENUM,VesselName,NodeName,matchvestype

#from .artlabel.gnn import ArtLabel

class iCafe:
	def __init__(self,path=None,config=None):
		if config is None:
			config = {}
		self.filename_solo = None
		#img series arrays
		self.I = {}
		#img series interp grid
		self.Iint = {}
		# rotation and translation matrix for multi contrast registration
		# key: seqname, value: 4*3 matrix, rotation matrix 3*3 + translation 1*3
		self.posRTMat = {}

		if path is not None:
			if not os.path.exists(path):
				raise FileNotFoundError('Path name not exist',path)
			if not os.path.isdir(path):
				raise FileNotFoundError('Result folder needed')
			self.setpath(path)
			self.settingfilename = os.path.join(self.path,'setting_TH_' + self.filename_solo + '.xml')
			self.xml =  XML(self.settingfilename)
			# allowed types for image loading
			#v: vesselness image
			#s: segmented image (probability map)
			#b: binary segmented image
			#i: region inside artery filled with id of snake
			#n: normalized image using Nyul
			#S10X: X=0-9, vts format multi-sequence image
			
			#always load 'o' first to load image size
			#self.I['o'] = None
			if 'I' in config:
				for k in config['I']:
					self.I[k] = None
			#load following images in I list
			self.loadImgs()
		else:
			print('Init with no path')
			self.path = None
			self.xml = XML()
		
		self.rzratio = 1
		if 'rzratio' in config:
			self.rzratio = config['rzratio']


		#img size
		self._tifimg = None
		self._SM = None
		self._SN = None
		self._SZ = None

		#model to predict landmark
		self.art_label_predictor  = None
		#predicted landmark
		self.pred_landmark = None

		#properties to be loaded when using
		self._snakelist = None
		self._swclist = None
		self._seeds = None

		self._veslist = None #list, first of vessel type, then each snake in that type
		self._veslist = None #list, first of vessel type, then each snake in that type
		self._vessnakelist = None #everything in one list

		self._cas = None
		self._reg = None

	def __repr__(self):
		if self.path is None:
			return 'iCafe init with no path'
		rpstr = 'Loaded ' + self.path
		if self._tifimg is not None:
			rpstr += ', shape of (%d,%d,%d)' % self._tifimg.shape
		loadedseq = list(self.I.keys())
		rpstr += ' With loaded seqs:' + ','.join(loadedseq)
		rpstr += ' Snakelist:%d' % (len(self.snakelist))
		return rpstr

	def __getitem__(self, key):
		return self.snakelist[key]

	@property
	def tifimg(self):
		if self._tifimg is None:
			self.loadImg('o')
		return self._tifimg

	# width (dim 1 of tif img)
	@property
	def SM(self):
		if self._SM is None:
			# SM SN SZ will be set from loading o
			self.loadImg('o')
		return self._SM

	# height (dim 0 of tif img)
	@property
	def SN(self):
		if self._SN is None:
			# SM SN SZ will be set from loading o
			self.loadImg('o')
		return self._SN

	# depth (dim 2 of tif img)
	@property
	def SZ(self):
		if self._SZ is None:
			# SM SN SZ will be set from loading o
			self.loadImg('o')
		return self._SZ

	@property
	def shape(self):
		return self.SM, self.SN, self.SZ

	@property
	def box(self):
		return 0, self.SM, 0, self.SN, 0, self.SZ

	@property
	def snakelist(self):
		if self._snakelist is None:
			self.loadSWC()
		return self._snakelist

	@snakelist.setter
	def snakelist(self,snakelist):
		self._snakelist = snakelist

	@property
	def NSnakes(self):
		return self.snakelist.NSnakes

	@property
	def swclist(self):
		if self._swclist is None:
			self.loadSWC()
		return self._swclist

	@swclist.setter
	def swclist(self,swclist):
		self._swclist = swclist

	@property
	def vessnakelist(self):
		if self._vessnakelist is None:
			self.loadVes()
		return self._vessnakelist

	@property
	def veslist(self):
		if self._veslist is None:
			self.loadVes()
		return self._veslist

	@property
	def cas(self):
		if self._cas is None:
			self._cas = CASCADE('E'+self.filename_solo.split('_')[1]+'_L',self.path+'/CASCADE')
		return self._cas

	@property
	def reg(self):
		if self._reg is None:
			self._reg = Reg(self.path, self.xml, self.datapath)
		return self._reg 
	
	@property
	def seeds(self):
		if self._seeds is None:
			self.loadSeeds()
		return self._seeds

	def setpath(self,path):
		self.path = path
		self.dbname = os.path.basename(os.path.abspath(path+'/..'))
		self.icafe_base_name = os.path.abspath(path+'/../../..').replace('\\','/')
		self.filename_solo = os.path.basename(path)
		if len(self.filename_solo.split('_'))==3:
			self.casename = self.filename_solo.split('_')[1]
		else:
			print('casename not following 0_XX_U')
		self.datapath = path.replace('result','data')
		if not os.path.exists(self.datapath):
			os.makedirs(self.datapath)
			print('mkdir',self.datapath)
		self.dcm_files = [i for i in os.listdir(self.datapath) if not os.path.isdir(self.datapath+'/'+i)]
		if len(self.dcm_files)>0:
			self.dcm_template = self.datapath + '/' + self.dcm_files[0]
		else:
			self.dcm_template = None

	def setDCMTemplate(self,dcm_path):
		shutil.copy(dcm_path, self.datapath + '/1.dcm')
		self.dcm_template = self.datapath + '/1.dcm'

	def getResFromDCMTemplate(self):
		if self.dcm_template is None:
			print('no dcm template')
			return
		dcm = pydicom.read_file(self.dcm_template)
		return dcm.PixelSpacing[0]

	def getPath(self,src):
		if src == 'o':
			return self.path+'/TH_'+self.filename_solo + '.tif'
		elif src in ['raw_ves','ves','seg_ves']:
			return self.path+'/tracing_'+src+'_TH_'+self.filename_solo+'.swc'
		elif len(src)==1 or src[0]=='S':
			return self.path+'/TH_'+self.filename_solo + src + '.tif'
		elif len(src)>4 and src[-4:]=='.swc':
			return self.path + '/tracing_' + src[:-4] + '_TH_' + self.filename_solo + '.swc'
		elif src[:5] == 'graph':
			return self.path + '/' + src + '_TH_' + self.filename_solo + '.pickle'
		elif len(src)>4 and src[-4:]=='.txt':
			return self.path + '/' + src[:-4] + '_TH_' + self.filename_solo + '.txt'
		else:
			return self.path+'/TH_'+self.filename_solo + src

	def existPath(self,src):
		return os.path.exists(self.getPath(src))

	def saveProjectAs(self, target_foler, db, img_srcs=['.tif', 'v.tif', 'h.tif'], swc_srcs=['raw_ves', 'ves', 'raw'], data=False):
		target_result_foler = target_foler + '/result/' + db + '/' + self.filename_solo
		if not os.path.exists(target_result_foler):
			os.makedirs(target_result_foler)
			print('mkdir', target_result_foler)
		target_data_foler = target_foler + '/data/' + db + '/' + self.filename_solo
		if not os.path.exists(target_data_foler):
			os.makedirs(target_data_foler)
			print('mkdir',target_data_foler)
		#dcm data
		if data:
			#complete copy of dicoms
			for d in self.dcm_files:
				shutil.copy(self.datapath +'/' + d, target_data_foler + '/' + d)
		else:
			if self.dcm_template is not None:
				shutil.copy(self.dcm_template, target_data_foler + '/' + os.path.basename(self.dcm_template))
		#saveas setting
		if os.path.exists(self.settingfilename):
			shutil.copy(self.settingfilename, target_result_foler + '/setting_TH_' + self.filename_solo + '.xml')
			print(self.filename_solo,'save as setting')

		#saveas images
		for srci in img_srcs:
			src_img_file = self.path+'/TH_'+self.filename_solo+srci
			if os.path.exists(src_img_file):
				shutil.copy(src_img_file, target_result_foler + '/TH_' + self.filename_solo + srci)
				print(self.filename_solo,'save as img',srci)

		#saveas traces
		for srci in swc_srcs:
			src_swc_file = self.path + '/tracing_' + srci +'_TH_' + self.filename_solo + '.swc'
			if os.path.exists(src_swc_file):
				shutil.copy(src_swc_file, target_result_foler + '/tracing_' + srci +'_TH_' + self.filename_solo + '.swc')
				print(self.filename_solo,'save as swc', srci)

		#seed
		src_file = self.path + '/seed_TH_' + self.filename_solo + '.txt'
		if os.path.exists(src_file):
			shutil.copy(src_file, target_result_foler + '/seed_TH_' + self.filename_solo + '.txt')
			print(self.filename_solo,'save as seeds')

		#percentile
		src_file = self.path + '/Per_TH_' + self.filename_solo + '.csv'
		if os.path.exists(src_file):
			shutil.copy(src_file, target_result_foler + '/Per_TH_' + self.filename_solo + '.csv')
			print(self.filename_solo,'save as per')

		#histogram
		src_file = self.path + '/hist_TH_' + self.filename_solo + '.txt'
		if os.path.exists(src_file):
			shutil.copy(src_file, target_result_foler + '/hist_TH_' + self.filename_solo + '.txt')
			print(self.filename_solo,'save as hist')

	##################
	#read image info
	##################
	from ._img import listAvailImgs, loadImgs, loadImg, _loadImgFile, _saveImgFile, saveImg, saveImgs, saveImgAs, loadVWI, getInt, getBoxInt, \
		getIntensityAlongSnake, getIntensityRaySnake, displaySlice, extractSlice, genSnakeMap, genArtMap, paintDistTransform, \
		searchDCM, createFromDCM, createFromImg,createFromVTS, normImg, imComputeInitForegroundModel, imComputeInitBackgroundModel

	##################
	#read/write trace info
	##################
	#swc
	from ._swc import loadSWC, writeSWC, readSnake
	# ves
	from ._swc import loadVes, _loadVesNoDuplicate, _loadVesNoChange, matchVesFromSnake
	
	##################
	#seeds 
	##################
	from ._seed import loadSeeds, addSeed, setSeeds, writeSeeds, clearSeeds, setSeedsSnakeList

	from ._ptnote import loadPtNote

	##################
	#graph representation
	##################
	#from .artlabel.gnn import generate_graph_icafe, read_pickle_graph, gnn_art_label
	from ._graph import generateGraph, writeGraph, readGraph, generateSimG


	##################
	#MPR generation
	##################
	from .interp.mpr import mpr,_generateMPRSnake,_generateCPRSnake,_generateCPPRSnake, mprStack, showMPRSnake
	from .interp.cs import cs, getCSImg, csStackRange, csStackNei, normPlane, getNormImg



	##################
	#Artery refinement
	##################
	# from .refine.refine import refSnake, refSnakeList
	# from .skeleton.skele import constructSkeleton, lowThresAddSkeleton
	# from .connect.snake import simpleRefSnake
	# from .connect.merge import mergeSnakeIntMatch,pathMatchInt

	# ##################
	# #VW Segmentation
	# ##################
	# from .vwseg.polarseg import polarVWSegArtery, polarVWSegCS
	# from .vwseg.measure import measureVWArt, saveMeasureResult, loadMeasureResult

