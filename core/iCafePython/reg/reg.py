import numpy as np
import os
import SimpleITK as sitk
import glob
from .reg_utils import calPermuteAxes, calFlipAxes, getRTMFromTransform, getTransformFromRTM, compositeRTM, metric_start_plot, metric_end_plot, metric_plot_values, metric_update_multires_iterations, readImgFromVts

class Reg:
	def __init__(self, path, xml, regdatapath):
		self.path = path
		self.xml = xml
		self.setRegDataPath(regdatapath)
		self.dcm_from_vts = False

	def setRegDataPath(self, regdatapath):
		if not os.path.exists(regdatapath):
			raise ValueError('Registration data path not exists')
		self.regdatapath = regdatapath
		# expect multi-contrast dcm organized in folders named 'S10X'
		reader = sitk.ImageSeriesReader()
		self.seqlist = {i:reader.GetGDCMSeriesFileNames(os.path.join(self.regdatapath, i)) for i in os.listdir(regdatapath) if os.path.isdir(regdatapath+'/'+i)}
		if not len(self.seqlist):
			#try loading using VTS format. All seq in the same folder with S10X format
			self.seqlist = {}
			for mra_seq in ['S10%d'%i for i in range(10)]:
				dcm_files_mra = glob.glob(self.regdatapath + '/*' + mra_seq + 'I*.dcm')
				if len(dcm_files_mra)==0:
					continue
				else:
					print('found seq',mra_seq,'with',len(dcm_files_mra),'dcms')
				dcm_files_mra.sort(key=lambda x: int(os.path.basename(x).split(mra_seq + 'I')[-1][:-4]))
				self.seqlist[mra_seq] = dcm_files_mra
			if len(self.seqlist):
				self.dcm_from_vts = True
		print('Available sequence list: ', self.seqlist.keys())

	def executeAll(self, ref_seq='S101', mra_seq='S104'):
		if ref_seq == mra_seq:
			raise ValueError('mra_seq as ref is not recommeneded')
		if ref_seq not in self.seqlist:
			raise ValueError('ref_seq not in available sequences')
		if mra_seq not in self.seqlist:
			raise ValueError('mra_seq not in available sequences')
		if len(self.seqlist) < 2:
			raise ValueError('Not enough available sequences')
		for seqi in self.seqlist:
			if seqi == ref_seq:
				continue
			print('='*20,'Registering',ref_seq, 'with', seqi, '='*20)
			if seqi == mra_seq:
				self.execute(ref_seq, seqi, update_MRA=True)
				print('MRA updated')
			else:
				self.execute(ref_seq, seqi)

	#if fixed_seq_img is set, do not read from seqlist, use fixed_seq_img directly as fixed_img
	def execute(self, fixed_seq, moving_seq, fixed_seq_src=None, resample=False, update_MRA=False, fixed_seq_img=None):
		self.prepareFixedImg(fixed_seq, fixed_seq_src, fixed_seq_img = fixed_seq_img)
		self.prepareMovingImg(moving_seq)
		self.rigidReg3D()
		self.xml.addSeqRTM(moving_seq, self.RTM)
		self.xml.writexml()
		if resample or update_MRA:
			self.saveResample(moving_seq, update_MRA)

	def prepareFixedImg(self, seq, src=None, fixed_seq_img=None):
		if src is None:
			if seq not in self.seqlist:
				raise ValueError(seq + ' not exists')
			self.fixed_RTM = self.xml.InitSeqRTM(seq)
		else:
			if seq not in src.seqlist:
				raise ValueError(seq + ' (ref src) not exists')
			self.fixed_RTM = src.xml.InitSeqRTM(seq)

		if fixed_seq_img is None:
			if self.dcm_from_vts:
				#itk cannot read directly from vts folder, because spacing in VTS is recorded differently
				img = readImgFromVts(self.seqlist[seq])
			else:
				reader = sitk.ImageSeriesReader()
				reader.SetFileNames(self.seqlist[seq])
				img = reader.Execute()
		else:
			# seq is itk img
			img = fixed_seq_img

		# save 'S10X_ori.tif'
		if src is None:
			tif_file_name = os.path.join(self.path, 'TH_' + os.path.basename(self.path) + seq + '_ori.tif')
			if not os.path.exists(tif_file_name):
				sitk.WriteImage(sitk.Cast(img, sitk.sitkInt16), tif_file_name)

		self.fixed_img = img


	def prepareMovingImg(self, seq):
		if seq not in self.seqlist:
			raise ValueError(seq + ' not exists')

		if self.dcm_from_vts:
			#itk cannot read directly from vts folder, because spacing in VTS is recorded differently
			img = readImgFromVts(self.seqlist[seq])
		else:
			reader = sitk.ImageSeriesReader()
			reader.SetFileNames(self.seqlist[seq])
			img = reader.Execute()

		permute = sitk.PermuteAxesImageFilter()
		permute.SetOrder(calPermuteAxes(self.fixed_img, img))
		img = permute.Execute(img)
		flip = sitk.FlipImageFilter()
		flip.SetFlipAxes(calFlipAxes(self.fixed_img, img))
		img = flip.Execute(img)

		# save 'S10X_ori.tif'
		tif_file_name = os.path.join(self.path, 'TH_' + os.path.basename(self.path) + seq + '_ori.tif')
		#if not os.path.exists(tif_file_name):
		sitk.WriteImage(sitk.Cast(img, sitk.sitkInt16), tif_file_name)

		self.moving_img = img

	def rigidReg3D(self):
		self.fixed_img.SetOrigin((0,0,0))
		self.fixed_img.SetDirection((1,0,0,0,1,0,0,0,1))
		self.moving_img.SetOrigin((0,0,0))
		self.moving_img.SetDirection((1,0,0,0,1,0,0,0,1))
		fixed_rez = self.fixed_img.GetSpacing()
		moving_rez = self.moving_img.GetSpacing()
		
		initialTx = sitk.CenteredTransformInitializer(sitk.Cast(self.fixed_img, sitk.sitkFloat32),
			sitk.Cast(self.moving_img, sitk.sitkFloat32),
			sitk.Euler3DTransform(),
			sitk.CenteredTransformInitializerFilter.GEOMETRY)
		reg_method = sitk.ImageRegistrationMethod()
		reg_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=256)
		reg_method.SetMetricSamplingStrategy(reg_method.RANDOM)
		reg_method.SetMetricSamplingPercentage(0.1)
		reg_method.SetInterpolator(sitk.sitkLinear)
		reg_method.SetOptimizerAsGradientDescent(learningRate=1e-4,
			numberOfIterations=300,
			convergenceWindowSize=15,
			estimateLearningRate=reg_method.EachIteration)
		reg_method.SetOptimizerScalesFromPhysicalShift()
		finalTx=sitk.Euler3DTransform(initialTx)
		reg_method.SetInitialTransform(finalTx)
		reg_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
		reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
		reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
		reg_method.AddCommand(sitk.sitkStartEvent,metric_start_plot)
		reg_method.AddCommand(sitk.sitkEndEvent,metric_end_plot)
		reg_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, 
			metric_update_multires_iterations) 
		reg_method.AddCommand(sitk.sitkIterationEvent, 
			lambda: metric_plot_values(reg_method))
		reg_method.Execute(sitk.Cast(self.fixed_img, sitk.sitkFloat32),
			sitk.Cast(self.moving_img, sitk.sitkFloat32))

		print('Registration complete!')
		print(finalTx)

		self.moving_RTM = getRTMFromTransform(finalTx, fixed_rez, moving_rez)
		self.RTM = compositeRTM(self.fixed_RTM, self.moving_RTM)

	def saveResample(self, moving_seq, update_MRA=False):
		fixed_rez = self.fixed_img.GetSpacing()
		moving_rez = self.moving_img.GetSpacing()
		finalTx = getTransformFromRTM(self.RTM, fixed_rez, moving_rez)

		resample = sitk.ResampleImageFilter()
		resample.SetReferenceImage(self.fixed_img)
		resample.SetInterpolator(sitk.sitkBSpline)  
		resample.SetTransform(finalTx)
		resample_img = resample.Execute(self.moving_img)

		# save 'S10X.tif'
		tif_file_name = os.path.join(self.path, 'TH_' + os.path.basename(self.path) + moving_seq + '.tif')
		sitk.WriteImage(sitk.Cast(resample_img, sitk.sitkInt16), tif_file_name)

		if update_MRA:
			# save '.tif'
			tif_file_name = os.path.join(self.path, 'TH_' + os.path.basename(self.path) + '.tif')
			sitk.WriteImage(sitk.Cast(resample_img, sitk.sitkInt16), tif_file_name)
