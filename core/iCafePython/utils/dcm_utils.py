import pydicom
import numpy as np
import cv2
import copy
from pydicom.pixel_data_handlers.numpy_handler import pack_bits

def exportDCM(dcm_template,cs_img,dcm_filename,ptidx,pxgap):
    pixelspacing = abs(float(dcm_template.PixelSpacing[0]))
    cs_img[cs_img < 0] = 0
    dcm_file = copy.deepcopy(dcm_template)
    dcm_file.SpacingBetweenSlices = str(pixelspacing * pxgap)
    dcm_file.SOPInstanceUID = dcm_file.SOPInstanceUID[:-len(dcm_file.SOPInstanceUID.split('.')[-1])] + '%d' % (ptidx + 1)
    dcm_file.ImagePositionPatient[2] = str(dcm_template.ImagePositionPatient[2] + ptidx * pxgap * pixelspacing)
    dcm_file.ImagePositionPatient[0] = str(dcm_template.ImagePositionPatient[0])
    dcm_file.ImagePositionPatient[1] = str(dcm_template.ImagePositionPatient[1])
    dcm_file.ImageOrientationPatient = ['1.00000', '0.00000', '0.00000', '0.00000', '1.00000', '0.00000']
    dcm_file.InstanceNumber = str(ptidx + 1)
    dcm_file.PixelData = cs_img.astype(np.uint16).tostring()
    dcm_file.Rows = cs_img.shape[0]
    dcm_file.Columns = cs_img.shape[1]
    #dcm_file.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  #uncompressed

    dcm_file.save_as(dcm_filename)

#dcm_folder contains %d
def exportDCMSeries(dcm_template_filename,img_stack,dcm_folder,id_offset=1,pxgap=1):
    dcm_template = pydicom.read_file(dcm_template_filename)
    if '%d' not in dcm_folder:
        raise ValueError('dcm_folder must contain %d')
    for di in range(img_stack.shape[2]):
        cs_img = img_stack[:,:,di]
        dcm_filename = dcm_folder%(di+id_offset)
        exportDCM(dcm_template,cs_img,dcm_filename,di+id_offset,pxgap)

def exportcsDCMforMOCHA(dcm_template,cs_img,dcm_filename,ptidx,pxgap,vesname,seq):
    pixelspacing = abs(float(dcm_template.PixelSpacing[0]))
    cs_img[cs_img < 0] = 0
    cs_img_pad = cv2.copyMakeBorder(cs_img, 260,260,260,260, borderType=cv2.BORDER_CONSTANT)
    dcm_file = copy.deepcopy(dcm_template)
    dcm_file.StudyDescription = vesname
    dcm_file.SeriesDescription = seq[-3:]
    dcm_file.SeriesNumber = seq[-3:]
    dcm_file.SOPInstanceUID = '0.'+seq[-1]+'.%d' % (ptidx+1)
    dcm_file.SOPInstanceUID = '0.'+seq[-1]
    dcm_file.StudyInstanceUID = '1'
    dcm_file.InstanceNumber = str(ptidx + 1)
    dcm_file.ImagePositionPatient[0] = str(dcm_file.ImagePositionPatient[0])
    dcm_file.ImagePositionPatient[1] = str(dcm_file.ImagePositionPatient[1])
    dcm_file.ImagePositionPatient[2] = str(dcm_file.ImagePositionPatient[2]+ptidx*pxgap*pixelspacing)
    dcm_file.ImageOrientationPatient = ['1.00000', '0.00000', '0.00000', '0.00000', '1.00000', '0.00000']
    dcm_file.SpacingBetweenSlices = str(pixelspacing * pxgap)
    dcm_file.PixelData = cs_img_pad.astype(np.uint16).tostring()
    dcm_file.Rows = cs_img_pad.shape[0]
    dcm_file.Columns = cs_img_pad.shape[1]
    
    cs_center = np.zeros(cs_img_pad.shape, dtype=np.uint8)
    cs_center[cs_img_pad.shape[0]//2, cs_img_pad.shape[1]//2] = 1
    cs_center[cs_img_pad.shape[0]//2, cs_img_pad.shape[1]//2] = 1
    #cs_center[cs_img_pad.shape[0]//2-1, cs_img_pad.shape[1]//2] = 1
    #cs_center[cs_img_pad.shape[0]//2+1, cs_img_pad.shape[1]//2] = 1
    #cs_center[cs_img_pad.shape[0]//2, cs_img_pad.shape[1]//2-1] = 1
    #cs_center[cs_img_pad.shape[0]//2, cs_img_pad.shape[1]//2+1] = 1
    dcm_file.add_new([0x6000,0x0010],'US',cs_img_pad.shape[0])
    dcm_file.add_new([0x6000,0x0011],'US',cs_img_pad.shape[1])
    dcm_file.add_new([0x6000,0x0015],'IS',1)
    dcm_file.add_new([0x6000,0x0040],'CS','G')
    dcm_file.add_new([0x6000,0x0050],'SS',[1,1])
    dcm_file.add_new([0x6000,0x0100],'US',1)
    dcm_file.add_new([0x6000,0x0102],'US',0)
    dcm_file.add_new([0x6000,0x3000],'OW',pack_bits(cs_center))

    dcm_file.save_as(dcm_filename)
    
#dcm_folder contains %d
def exportcsDCMSeriesforMOCHA(dcm_template_filename,img_stack,dcm_folder,vesname,seq,id_offset=1,pxgap=1):
    dcm_template = pydicom.read_file(dcm_template_filename)
    if '%d' not in dcm_folder:
        raise ValueError('dcm_folder must contain %d')
    for di in range(img_stack.shape[2]):
        cs_img = img_stack[:,:,di]
        dcm_filename = dcm_folder%(di+id_offset)
        exportcsDCMforMOCHA(dcm_template,cs_img,dcm_filename,di+id_offset,pxgap,vesname,seq)
    
def exportmprDCMforMOCHA(dcm_template,mpr_img,dcm_filename,rot,pxgap,vesname,seq):
    pixelspacing = abs(float(dcm_template.PixelSpacing[0]))
    dcm_file = copy.deepcopy(dcm_template)
    dcm_file.StudyDescription = vesname
    dcm_file.SeriesDescription = str(int(seq[-3:])+100)
    dcm_file.SeriesNumber = str(int(seq[-3:])+100)
    dcm_file.SOPInstanceUID = '1.'+seq[-1]+'.%d' % (rot+1)
    dcm_file.SeriesInstanceUID = '1.'+seq[-1]
    dcm_file.StudyInstanceUID = '1'
    dcm_file.InstanceNumber = str(rot+1)
    dcm_file.ImagePositionPatient[0] = str(dcm_file.ImagePositionPatient[0])
    dcm_file.ImagePositionPatient[1] = str(dcm_file.ImagePositionPatient[1]+(rot+1)*pixelspacing)
    dcm_file.ImagePositionPatient[2] = str(dcm_file.ImagePositionPatient[2])
    dcm_file.ImageOrientationPatient = ['1.00000', '0.00000', '0.00000', '0.00000', '0.00000', '1.00000']
    dcm_file.SpacingBetweenSlices = str(pixelspacing * pxgap)
    dcm_file.PixelData = mpr_img.astype(np.uint16).tostring()
    dcm_file.Rows = mpr_img.shape[0]
    dcm_file.Columns = mpr_img.shape[1]
            
    mpr_center = np.zeros(mpr_img.shape, dtype=np.uint8)
    mpr_center[::8, mpr_img.shape[1]//2] = 1
    dcm_file.add_new([0x6000,0x0010],'US',mpr_img.shape[0])
    dcm_file.add_new([0x6000,0x0011],'US',mpr_img.shape[1])
    dcm_file.add_new([0x6000,0x0015],'IS',1)
    dcm_file.add_new([0x6000,0x0040],'CS','G')
    dcm_file.add_new([0x6000,0x0050],'SS',[1,1])
    dcm_file.add_new([0x6000,0x0100],'US',1)
    dcm_file.add_new([0x6000,0x0102],'US',0)
    dcm_file.add_new([0x6000,0x3000],'OW',pack_bits(mpr_center))

    dcm_file.save_as(dcm_filename)  

def exportmprDCMSeriesforMOCHA(dcm_template_filename,img_stack,dcm_folder,vesname,seq,pxgap=1):
    dcm_template = pydicom.read_file(dcm_template_filename)
    if '%d' not in dcm_folder:
        raise ValueError('dcm_folder must contain %d')
    for roti in range(img_stack.shape[2]):
        mpr_img = img_stack[:,:,roti]
        dcm_filename = dcm_folder%((roti)*3)
        exportmprDCMforMOCHA(dcm_template,mpr_img,dcm_filename,roti,pxgap,vesname,seq)