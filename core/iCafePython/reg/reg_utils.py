import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom
# from IPython.display import clear_output

# sitk image object


def calPermuteAxes(img1, img2):
	dir1, sign1 = calImageOrientation(img1)
	dir2, sign2 = calImageOrientation(img2)

	permuteAxes = [-1, -1, -1]
	for i in range(3):
		for j in range(3):
			if dir2[j] == dir1[i]:
				permuteAxes[i] = j
				continue
	return permuteAxes


def calFlipAxes(img1, img2):
	dir1, sign1 = calImageOrientation(img1)
	dir2, sign2 = calImageOrientation(img2)

	flipAxes = [False, False, False]
	for i in range(3):
		if sign1[i] * sign2[i] < 0:
			flipAxes[i] = True
	return flipAxes


def calImageOrientation(img):
	d = img.GetDirection()
	xdir = np.array([d[0], d[3], d[6]])
	ydir = np.array([d[1], d[4], d[7]])
	zdir = np.array([d[2], d[5], d[8]])
	imgdir = [np.argmax(abs(xdir)), np.argmax(abs(ydir)), np.argmax(abs(zdir))]
	imgdirsign = [
		np.sign(xdir[np.argmax(abs(xdir))]),
		np.sign(ydir[np.argmax(abs(ydir))]),
		np.sign(zdir[np.argmax(abs(zdir))])
	]
	return imgdir, imgdirsign


def getRTMFromTransform(Tx, fixed_rez, moving_rez):
	# RTM in pixel unit
	c = np.array(Tx.GetCenter())
	A = np.resize(Tx.GetMatrix(), (3, 3))
	t = np.array(Tx.GetTranslation())
	o = t + c - np.matmul(A, c)

	A00 = A[0, 0] * fixed_rez[0] / moving_rez[0]
	A01 = A[0, 1] * fixed_rez[1] / moving_rez[0]
	A02 = A[0, 2] * fixed_rez[2] / moving_rez[0]
	A10 = A[1, 0] * fixed_rez[0] / moving_rez[1]
	A11 = A[1, 1] * fixed_rez[1] / moving_rez[1]
	A12 = A[1, 2] * fixed_rez[2] / moving_rez[1]
	A20 = A[2, 0] * fixed_rez[0] / moving_rez[2]
	A21 = A[2, 1] * fixed_rez[1] / moving_rez[2]
	A22 = A[2, 2] * fixed_rez[2] / moving_rez[2]
	o0 = o[0] / moving_rez[0]
	o1 = o[1] / moving_rez[1]
	o2 = o[2] / moving_rez[2]

	rtm = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22], [o0, o1, o2]]

	return rtm


def getTransformFromRTM(rtm, fixed_rez, moving_rez):
	# rotation and translation in physical unit
	A00p = rtm[0][0] * moving_rez[0] / fixed_rez[0]
	A01p = rtm[0][1] * moving_rez[0] / fixed_rez[1]
	A02p = rtm[0][2] * moving_rez[0] / fixed_rez[2]
	A10p = rtm[1][0] * moving_rez[1] / fixed_rez[0]
	A11p = rtm[1][1] * moving_rez[1] / fixed_rez[1]
	A12p = rtm[1][2] * moving_rez[1] / fixed_rez[2]
	A20p = rtm[2][0] * moving_rez[2] / fixed_rez[0]
	A21p = rtm[2][1] * moving_rez[2] / fixed_rez[1]
	A22p = rtm[2][2] * moving_rez[2] / fixed_rez[2]
	o0p = rtm[3][0] * moving_rez[0]
	o1p = rtm[3][1] * moving_rez[1]
	o2p = rtm[3][2] * moving_rez[2]

	Tx = sitk.AffineTransform(3)
	Tx.SetMatrix([A00p, A01p, A02p, A10p, A11p, A12p, A20p, A21p, A22p])
	Tx.SetTranslation([o0p, o1p, o2p])
	Tx.SetCenter([0, 0, 0])

	return Tx


def compositeRTM(RTM0, RTM1):
	# T(x) = T1(T0(x)) = Ax + o = A1A0x + (A1o0 + o1)
	A0 = np.resize(RTM0[0:3], (3, 3))
	A1 = np.resize(RTM1[0:3], (3, 3))
	o0 = np.array(RTM0[3])
	o1 = np.array(RTM1[3])

	A = np.matmul(A1, A0)
	o = np.matmul(A1, o0) + o1

	rtm = [[A[0, 0], A[0, 1], A[0, 2]], [A[1, 0], A[1, 1], A[1, 2]], [A[2, 0], A[2, 1], A[2, 2]], [o[0], o[1], o[2]]]

	return rtm


# Callback we associate with the StartEvent, sets up our new data.
def metric_start_plot():
	global metric_values, multires_iterations
	global current_iteration_number

	metric_values = []
	multires_iterations = []
	current_iteration_number = -1


# Callback we associate with the EndEvent, do cleanup of data and figure.
def metric_end_plot():
	global metric_values, multires_iterations
	global current_iteration_number

	del metric_values
	del multires_iterations
	del current_iteration_number
	# Close figure, we don't want to get a duplicate of the plot latter on
	plt.close()


# Callback we associate with the IterationEvent, update our data and display
# new figure.
def metric_plot_values(registration_method):
	global metric_values, multires_iterations
	global current_iteration_number

	# Some optimizers report an iteration event for function evaluations and not
	# a complete iteration, we only want to update every iteration.
	if registration_method.GetOptimizerIteration() == current_iteration_number:
		return

	current_iteration_number = registration_method.GetOptimizerIteration()
	metric_values.append(registration_method.GetMetricValue())
	# Clear the output area (wait=True, to reduce flickering), and plot
	# current data.
	clear_output(wait=True)
	# Plot the similarity metric values.
	plt.plot(metric_values, 'r')
	plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
	plt.xlabel('Iteration Number', fontsize=12)
	plt.ylabel('Metric Value', fontsize=12)
	plt.show()


# Callback we associate with the MultiResolutionIterationEvent, update the
# index into the metric_values list.
def metric_update_multires_iterations():
	global metric_values, multires_iterations
	multires_iterations.append(len(metric_values))


# vts slice thickness is different. need to read slice by slice and set spacing manually
def readImgFromVts(dcm_files):
	dcm = pydicom.read_file(dcm_files[0])
	img = np.zeros((len(dcm_files), dcm.Rows, dcm.Columns))
	for slicei in range(len(dcm_files)):
		dcm_filename = dcm_files[slicei]
		dcm = pydicom.read_file(dcm_filename)
		img[slicei] = dcm.pixel_array
	simg = sitk.GetImageFromArray(img)
	spacing = simg.GetSpacing()
	pixel_spacing = dcm.PixelSpacing
	slice_thickness = float(dcm.SliceThickness) #float(dcm.SpacingBetweenSlices) +
	corrected_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), slice_thickness)
	print('spacing', corrected_spacing)
	simg.SetSpacing(corrected_spacing)
	return simg


def norm_density(x, mu, sigma):
	epsilion = 1.192e-7 # np.finfo(float).eps
	dcm = pydicom.read_file(dcm_files[0])
	img = np.zeros((len(dcm_files),dcm.Rows, dcm.Columns))
	for slicei in range(len(dcm_files)):
		dcm_filename = dcm_files[slicei]
		dcm = pydicom.read_file(dcm_filename)
		img[slicei] = dcm.pixel_array
	simg = sitk.GetImageFromArray(img)
	spacing = simg.GetSpacing()
	pixel_spacing = dcm.PixelSpacing
	slice_thickness =  float(dcm.SliceThickness) #float(dcm.SpacingBetweenSlices) +
	corrected_spacing = (float(pixel_spacing[0]),float(pixel_spacing[1]),slice_thickness)
	print('spacing',corrected_spacing)
	simg.SetSpacing(corrected_spacing)
	return simg

def norm_density(x, mu, sigma):
	epsilion = 1.192e-7  # np.finfo(float).eps
	p = 1 / max(sigma * np.sqrt(2 * np.pi), epsilion) * np.exp(-pow(x - mu, 2) / max(2 * pow(sigma, 2), epsilion))
	return p