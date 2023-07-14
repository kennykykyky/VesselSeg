import numpy as np
import SimpleITK as sitk
from ..point3d import Point3D
import math
from scipy.interpolate import RegularGridInterpolator
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import eval_linear, filter_cubic
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from .crop_utils import croppatch

def expandImg(oriimg,cropregion,targetshape):
	if cropregion[2]==-1:
		cropregion[2] = targetshape[0]
	if cropregion[3]==-1:
		cropregion[3] = targetshape[1]
	if oriimg.shape[0]>cropregion[2]-cropregion[0]:
		print('oriimg shape 0 larger than crop region')
		oriimg = oriimg[:cropregion[2]-cropregion[0]]
	if oriimg.shape[1]>cropregion[3]-cropregion[1]:
		print('oriimg shape 1 larger than crop region')
		oriimg = oriimg[:,:cropregion[3]-cropregion[1]]
	expandedimg = np.zeros(targetshape)
	expandedimg[cropregion[0]:cropregion[0]+oriimg.shape[0],cropregion[1]:cropregion[1]+oriimg.shape[1]] = oriimg
	return expandedimg

def resample3d(tiffilename,spacingbetweenslices,pixelspacing=1):
	#iso resample
	oritif_image = sitk.ReadImage(tiffilename)
	resample = sitk.ResampleImageFilter()
	resample.SetOutputDirection = oritif_image.GetDirection()
	resample.SetOutputOrigin = oritif_image.GetOrigin()
	new_spacing = list(oritif_image.GetSpacing())
	new_spacing[2] = new_spacing[2]* pixelspacing/spacingbetweenslices
	print('new_spacing',new_spacing)
	resample.SetOutputSpacing(new_spacing)
	orig_size = np.array(oritif_image.GetSize(), dtype=np.int)
	new_size = orig_size
	new_size[2] = int(round(new_size[2]*(spacingbetweenslices/pixelspacing)))
	new_size = [int(s) for s in new_size]
	print('new size',new_size)
	resample.SetSize(new_size)
	resample.SetInterpolator(sitk.sitkBSpline)
	resampledimg = resample.Execute(oritif_image)
	sitk.WriteImage(sitk.Cast(resampledimg,sitk.sitkInt16),tiffilename)

def resamplexyzPlane(tiffilename, xZoomRatio = 1, yZoomRatio = 1, zZoomRatio = 1):
	# Resample image (Coverage remains the same but resolution will change) - KY
	oritif_image = sitk.ReadImage(tiffilename)
	resample = sitk.ResampleImageFilter()
	resample.SetOutputDirection = oritif_image.GetDirection()
	resample.SetOutputOrigin = oritif_image.GetOrigin()
	new_spacing = list(oritif_image.GetSpacing())
	new_spacing[0] = new_spacing[0]* xZoomRatio
	new_spacing[1] = new_spacing[1]* yZoomRatio
	new_spacing[2] = new_spacing[2]* zZoomRatio
	print('new_spacing',new_spacing)
	resample.SetOutputSpacing(new_spacing)
	orig_size = np.array(oritif_image.GetSize(), dtype=np.int)
	new_size = orig_size
	new_size[0] = int(round(new_size[0] / xZoomRatio))
	new_size[1] = int(round(new_size[1] / yZoomRatio))
	new_size[1] = int(round(new_size[1] / zZoomRatio))
	new_size = [int(s) for s in new_size]
	print('new size',new_size)
	resample.SetSize(new_size)
	resample.SetInterpolator(sitk.sitkBSpline)
	resampledimg = resample.Execute(oritif_image)
	sitk.WriteImage(sitk.Cast(resampledimg,sitk.sitkInt16),tiffilename)

def rtTransform(pos, rtm):
	ix = pos.x
	iy = pos.y
	iz = pos.z
	ax = rtm[0][0] * ix + rtm[0][1] * iy + rtm[0][2] * iz + rtm[3][0]
	ay = rtm[1][0] * ix + rtm[1][1] * iy + rtm[1][2] * iz + rtm[3][1]
	az = rtm[2][0] * ix + rtm[2][1] * iy + rtm[2][2] * iz + rtm[3][2]
	return Point3D(ax, ay, az)


def topolar(car_img, rsamples=0, thsamples=180, intmethod='linear'):
	# BUG in cubic
	if rsamples == 0:
		rsamples = car_img.shape[0] // 2
	if len(car_img.shape) == 2:
		cimg = car_img[:, :, None]
	elif len(car_img.shape) == 3:
		cimg = car_img
	else:
		print('channel not 2/3')
		return

	SUBTH = 360 / thsamples
	height, width, channel = cimg.shape

	grid = UCGrid((0, cimg.shape[1] - 1, cimg.shape[1]), (0, cimg.shape[0] - 1, cimg.shape[0]))

	# filter values
	if intmethod == 'cubic':
		coeffs = filter_cubic(grid, cimg)

	rth = np.zeros((thsamples, rsamples, channel))
	for th in range(thsamples):
		for r in range(rsamples):
			inty = cimg.shape[0] // 2 + r * math.sin(th * SUBTH / 180 * math.pi)
			intx = cimg.shape[1] // 2 + r * math.cos(th * SUBTH / 180 * math.pi)
			if intx >= cimg.shape[1] - 1 or inty >= cimg.shape[0] - 1:
				rth[th, r] = 0
			elif intx < 0 or inty < 0:
				rth[th, r] = 0
			else:
				if intmethod == 'cubic':
					rth[th, r] = eval_cubic(grid, coeffs, np.array([inty, intx]))
				elif intmethod == 'linear':
					rth[th, r] = eval_linear(grid, cimg, np.array([inty, intx]))

	if len(car_img.shape) == 2:
		return rth[:, :, 0]
	else:
		return rth


def tocart(polar_img, rheight=0, rwidth=0, intmethod='linear'):
	if rheight == 0 or rwidth == 0:
		rheight = polar_img.shape[1] * 2
		rwidth = polar_img.shape[1] * 2
	# transfer to cartesian based
	if len(polar_img.shape) == 2:
		cimg = polar_img[:, :, None]
	elif len(polar_img.shape) == 3:
		cimg = polar_img
	else:
		print('channel not 2/3')
		return
	rth, rr, rchannel = cimg.shape
	SUBTH = 360 / rth

	grid = UCGrid((0, cimg.shape[0] - 1, cimg.shape[0]), (0, cimg.shape[1] - 1, cimg.shape[1]))

	test_out_c = np.zeros((rheight, rwidth, rchannel))
	for h in range(rheight):
		for w in range(rwidth):
			hy = h - rheight // 2
			wx = w - rwidth // 2
			intradius = int(np.sqrt(hy * hy + wx * wx))
			cth = math.atan2(hy, wx) / np.pi * 180
			if cth < 0:
				intth = (360 + cth)
			else:
				intth = (cth)

			intth /= SUBTH
			if intth > cimg.shape[0] - 1:
				intth = cimg.shape[0] - 1
			# print(intradius,intth)
			if intradius >= cimg.shape[1]:
				test_out_c[h, w] = 0
			else:
				test_out_c[h, w] = eval_linear(grid, cimg, np.array([intth, intradius]))
	if len(polar_img.shape) == 2:
		return test_out_c[:, :, 0]
	else:
		return test_out_c

def get_grad_img(polar_patch):
    OFFX = 2
    polar_patch_gaussian = gaussian_filter(polar_patch, sigma=5)
    sy = ndimage.sobel(polar_patch_gaussian, axis=0, mode='constant')
    sx = ndimage.sobel(polar_patch_gaussian, axis=1, mode='constant')
    polar_grad = np.hypot(sx, sy)
    polar_grad = croppatch(croppatch(polar_grad, 128, 128, 127, 127), 127, 127-OFFX, 128, 128)
    polar_grad[0] = polar_grad[1]
    polar_grad[-1] = polar_grad[-2]
    gradimg = polar_grad #/ np.max(polar_grad)
    return gradimg


# paint within radius at ct position
def paint_dist(img_fill, ct, rad, target=1):
	ct_int = ct.intlst()
	rad_int = int(round(rad))
	img_fill[tuple(ct_int)] = target
	for ofx in range(-rad_int, rad_int):
		for ofy in range(-rad_int, rad_int):
			for ofz in range(-rad_int, rad_int):
				cpos = Point3D([ct_int[0] + ofx, ct_int[1] + ofy, ct_int[2] + ofz])
				cpos.boundList(img_fill.shape)
				cdist = cpos.dist(ct)
				if cdist > rad:
					continue
				img_fill[tuple(cpos.intlst())] = target

#paint only unique labels, skip for those pos covered by 2+ snake
def paint_dist_unique(img_fill, ct, rad, target=1,markduplicate=1):
	ct_int = ct.intlst()
	rad_int = int(round(rad))
	img_fill[tuple(ct_int)] = target
	for ofx in range(-rad_int, rad_int+1):
		for ofy in range(-rad_int, rad_int+1):
			for ofz in range(-rad_int, rad_int+1):
				cpos = Point3D([ct_int[0] + ofx, ct_int[1] + ofy, ct_int[2] + ofz])
				cpos.boundList(img_fill.shape)
				cdist = cpos.dist(ct)
				if cdist > rad:
					continue
				if img_fill[tuple(cpos.intlst())] == 0:
					img_fill[tuple(cpos.intlst())] = target
				elif img_fill[tuple(cpos.intlst())] == target:
					continue
				else:
					if markduplicate:
						#set duplicate voxel as -1
						if img_fill[tuple(cpos.intlst())] != target:
							img_fill[tuple(cpos.intlst())] = -1
					else:
						#set duplicate voxel as the new val
						img_fill[tuple(cpos.intlst())] = target


def paint_dist_transform(img_fill,ct,rad):
    ct_int = ct.intlst()
    rad_int = int(np.ceil(rad))
    for ofx in range(-rad_int,rad_int):
        for ofy in range(-rad_int,rad_int):
            for ofz in range(-rad_int,rad_int):
                cpos = Point3D([ct_int[0]+ofx, ct_int[1]+ofy, ct_int[2]+ofz])
                cpos.boundList(img_fill.shape)
                if ofx==0 and ofy==0 and ofz==0:
                    img_fill[tuple(cpos.intlst())][0] = 1
                    img_fill[tuple(cpos.intlst())][1:4] = (cpos-ct).lst()
                    continue
                cdist = cpos.dist(ct)
                if cdist>rad:
                    continue
                if (rad-cdist)/rad>img_fill[tuple(cpos.intlst())][0]:
                    img_fill[tuple(cpos.intlst())][0] = (rad-cdist)/rad
                    img_fill[tuple(cpos.intlst())][1:4] = (cpos-ct).lst()

import cc3d
#find centroid for each connected region in foreground
#labels_in is a binary 3D image
def connectedCentroid(labels_in):
    labels_out = cc3d.connected_components(labels_in)
    N = np.max(labels_out)
    con_seeds = []
    for labeli in range(1,N+1):
        pts = [Point3D(i) for i in np.argwhere(labels_out == labeli)]
        print('\r','labeli',labeli,'/',N,'with',len(pts),'pixels',end='')
		#find the pt with mean min distance to all the other pts
        mdst = []
        for i in range(len(pts)):
            mdst.append(np.mean([pts[i].dist(pts[j]) for j in range(len(pts))]))
        minid = np.argmin(mdst)
        con_seeds.append(pts[minid].lst())
    return con_seeds
