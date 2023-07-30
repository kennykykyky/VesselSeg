import xml.etree.ElementTree as ET
import numpy as np
import os
import glob
import pydicom
from ..utils.dcm_utils import exportDCMSeries

class CASCADE:
    def __init__(self, QVJname, QVJdir, dcmdir=None, side='L', forceinit=0):
        self.ExamID = QVJname[:-2]
        self.side = side
        self.QVJdir = QVJdir
        if not os.path.exists(QVJdir):
            os.mkdir(QVJdir)
        if dcmdir is None:
            self.dcmdir = QVJdir
        else:
            self.dcmdir = dcmdir
        dcmfiles = glob.glob(self.dcmdir + '/*.dcm')

        if len(dcmfiles):
            self.mddcm = pydicom.read_file(dcmfiles[len(dcmfiles) // 2])
            self.dcmsz = self.mddcm.pixel_array.shape[0]
        self.qvsroots = []
        self.qvsnames = []
        self.HTML_NS = "vil.rad.washington.edu"
        self.QVJfilename = self.ExamID + '_' + self.side + '.QVJ'
        # self.ns = {'ns': 'vil.rad.washington.edu'}
        # ET.register_namespace("", "vil.rad.washington.edu")

        QVJpath = os.path.join(QVJdir, self.ExamID + '.QVJ')
        print(QVJpath)
        if os.path.exists(QVJpath) and forceinit == 0:
            print('Loading QVJ', QVJpath)
            self.qvjroot = ET.parse(QVJpath).getroot()
            # slice id for dcm files
            self.sliceid = 0
            QVSpaths = os.path.join(QVJdir, self.ExamID + 'S*_' + self.side + '.QVS')
            QVSlists = glob.glob(QVSpaths)
            if len(QVSlists) == 0:
                print('No .QVS found in the specified dir!')
            else:
                for qvsi in QVSlists:
                    self.addQVSfile(qvsi)
                    #print('Adding QVS', os.path.basename(qvsi))
        else:
            self.initQVJ()

    def addQVSfile(self, QVSfile):
        qvsroot = ET.parse(QVSfile).getroot()
        if self.sliceid == 0:
            self.sliceid += len(qvsroot.findall('QVAS_Image'))
            print('add image slices, sliceid',self.sliceid)
        self.qvsnames.append(QVSfile)
        self.qvsroots.append(qvsroot)

    def initQVJ(self):
        # qvjroot = ET.Element(ET.QName(self.HTML_NS,'QVAS_Project'))
        qvjroot = ET.Element('QVAS_Project')
        qvjroot.set("xmlns", "vil.rad.washington.edu")
        QVAS_Version = ET.SubElement(qvjroot, 'QVAS_Version')
        QVAS_Version.set("xmlns", "")
        QVAS_Version.text = "1.0"
        LastUsername = ET.SubElement(qvjroot, 'LastUsername')
        LastUsername.set("xmlns", "")
        QVAS_System_Info = ET.SubElement(qvjroot, 'QVAS_System_Info')
        QVAS_System_Info.set("xmlns", "")
        AnalysisMode = ET.SubElement(QVAS_System_Info, 'AnalysisMode')
        AnalysisMode.text = "1"
        ImageLocationStatus = ET.SubElement(QVAS_System_Info, 'ImageLocationStatus')
        ImageLocationStatus.text = "0"
        QVAS_Current_Series_List = ET.SubElement(qvjroot, 'QVAS_Current_Series_List')
        CurrentSeriesName = ET.SubElement(QVAS_Current_Series_List, 'CurrentSeriesName')
        ROI = ET.SubElement(QVAS_Current_Series_List, 'ROI')
        ROI.set("x1", "0")
        ROI.set("x2", "511")
        ROI.set("y1", "0")
        ROI.set("y2", "511")
        QVAS_Loaded_Series_List = ET.SubElement(qvjroot, 'QVAS_Loaded_Series_List')
        QVAS_Loaded_Series_List.set("xmlns", "")

        Location_Comment = ET.SubElement(qvjroot, 'Location_Comment')
        Location_Comment.set("xmlns", "")
        Project_Comments = ET.SubElement(qvjroot, 'Project_Comments')
        Project_Comments.set("xmlns", "")
        self.qvjroot = qvjroot
        # slice id for dcm files
        self.sliceid = 0
        print('Init QVJ')

    def clearQVS(self,QVSID):
        if type(QVSID) == int:
            QVSID = str(QVSID)
        for i in range(len(self.qvsnames)):
            if os.path.basename(self.qvsnames[i])[-9:-6] == QVSID:
                print('clear Seq',QVSID)
                del self.qvsnames[i]
                del self.qvsroots[i]
                break

    def initQVS(self, QVSID):
        if type(QVSID) == int:
            QVSID = str(QVSID)
        availqvs = [os.path.basename(i)[-9:-6] for i in self.qvsnames]
        if QVSID in availqvs:
            print('Seq exist')
            return
        QVSName = self.ExamID + 'S' + QVSID + '_' + self.side + '.QVS'
        QVSpath = os.path.join(self.QVJdir, QVSName)
        qvsroot = ET.Element('QVAS_Series')
        qvsroot.set("xmlns", "vil.rad.washington.edu")
        QVAS_Version = ET.SubElement(qvsroot, 'QVAS_Version')
        QVAS_Version.set("xmlns", "")
        QVAS_Version.text = "1.0"
        QVAS_Series_Info = ET.SubElement(qvsroot, 'QVAS_Series_Info')
        QVAS_Series_Info.set("xmlns", "")
        SeriesName = ET.SubElement(QVAS_Series_Info, 'SeriesName')
        SeriesName.text = self.ExamID + 'S' + QVSID + '_' + self.side
        SeriesPath = ET.SubElement(QVAS_Series_Info, 'SeriesPath')
        SeriesDescription = ET.SubElement(QVAS_Series_Info, 'SeriesDescription')
        SeriesLevel = ET.SubElement(QVAS_Series_Info, 'SeriesLevel')
        SeriesLevel.text = '-999'
        SeriesWindow = ET.SubElement(QVAS_Series_Info, 'SeriesWindow')
        SeriesWindow.text = '1000'
        # SNAP_Para_k2 = ET.SubElement(QVAS_Series_Info, 'SNAP_Para_k2')
        # SNAP_Para_k2.text = '1.000000'
        # shiftAfterPixelSizeAdjustment_X = ET.SubElement(QVAS_Series_Info, 'shiftAfterPixelSizeAdjustment_X')
        # shiftAfterPixelSizeAdjustment_X.text = '0'
        # shiftAfterPixelSizeAdjustment_Y = ET.SubElement(QVAS_Series_Info, 'shiftAfterPixelSizeAdjustment_Y')
        # shiftAfterPixelSizeAdjustment_Y.text = '0'
        # FOV_Adjusted = ET.SubElement(QVAS_Series_Info, 'FOV_Adjusted')
        # FOV_Adjusted.text = '1'

        # DCMpath = os.path.join(self.dcmdir,self.ExamID+'S'+QVSID+'I*.dcm')
        # DCMlists = glob.glob(DCMpath)


        self.qvsnames.append(QVSpath)
        self.qvsroots.append(qvsroot)
        print('QVS', QVSID, 'added')
        self.setCurrentSeries(QVSID)
        self.refreshLoadedSeries()

    # only for single snake. Will only use first last slice as segstart/end
    def refreshSeqImage(self,QVSID):
        qvsroot = None
        for si in range(len(self.qvsnames)):
            if os.path.basename(self.qvsnames[si])[-9:-6] == QVSID:
                qvsroot = self.qvsroots[si]
                break
        assert qvsroot is not None

        DCMlists = glob.glob(self.dcmdir+'/'+self.ExamID+'S'+QVSID+'I*.dcm')
        if len(DCMlists) == 0:
            print('No DCM for S' + QVSID,self.dcmdir+'/'+self.ExamID+'S'+QVSID+'I*.dcm')
            return
        DCMlists.sort(key=lambda x: int(x.split('S' + QVSID)[-1][1:-4]))
        for di in range(len(DCMlists)):
            dcmbasename = DCMlists[di]
            if di == 0:
                image_description = 'SegStart'
            elif di == len(DCMlists) - 1:
                image_description = 'SegEnd'
            else:
                image_description = None
            self.addQVASImage(qvsroot, dcmbasename, image_description)
        return np.arange(1,len(DCMlists)+1).tolist()

    # export dcm and add node in qvsroot
    def addSeqImage(self,QVSID,img_stack,dcm_template):
        if type(QVSID) == int:
            QVSID = str(QVSID)
        availqvs = [os.path.basename(i)[-9:-6] for i in self.qvsnames]
        if QVSID not in availqvs:
            print('Init Seq')
            self.initQVS(QVSID)

        #export dicom images
        dcm_name = self.ExamID+'S'+QVSID+'I%d.dcm'
        dcm_folder = self.dcmdir+'/'+dcm_name
        pxgap = 1
        exportDCMSeries(dcm_template, img_stack, dcm_folder, self.sliceid+1, pxgap)

        qvsroot = None
        for si in range(len(self.qvsnames)):
            if os.path.basename(self.qvsnames[si])[-9:-6]==QVSID:
                qvsroot = self.qvsroots[si]
                break
        assert qvsroot is not None

        img_ids = []
        for di in range(img_stack.shape[2]):
            self.sliceid += 1
            img_ids.append(self.sliceid)
            dcmbasename = self.ExamID + 'S' + QVSID + 'I' + str(self.sliceid)
            if di==0:
                image_description = 'SegStart'
            elif di==img_stack.shape[2]-1:
                image_description = 'SegEnd'
            else:
                image_description = None
            self.addQVASImage(qvsroot,dcmbasename,image_description)
        return img_ids

    #add child node of QVAS_Image to qvsroot
    def addQVASImage(self,qvsroot,dcmbasename,image_description):
        QVAS_Image = ET.SubElement(qvsroot, 'QVAS_Image')
        QVAS_Image.set("xmlns", "")
        QVAS_Image.set("ImageName", dcmbasename)
        Translation = ET.SubElement(QVAS_Image, 'Translation')
        Rotation = ET.SubElement(Translation, 'Rotation')
        Angle = ET.SubElement(Translation, 'Angle')
        Angle.text = '0.00'
        Point = ET.SubElement(Translation, 'Point')
        Point.set("y", "0.0")
        Point.set("x", "0.0")
        ShiftAfterRotation = ET.SubElement(Translation, 'ShiftAfterRotation ')
        ShiftAfterRotation.set("y", "0.00")
        ShiftAfterRotation.set("x", "0.00")

        ImageFilePath = ET.SubElement(QVAS_Image, 'ImageFilePath')
        ImageFilePath.text = dcmbasename + '.dcm'
        ImageDescription = ET.SubElement(QVAS_Image, 'ImageDescription')
        if image_description is not None:
            ImageDescription.text = image_description
        ImageMode = ET.SubElement(QVAS_Image, 'ImageMode')
        ImageBifurcationLevel = ET.SubElement(QVAS_Image, 'ImageBifurcationLevel')
        ImageBifurcationLevel.text = '-999'

    def setCurrentSeries(self, QVSID):
        if type(QVSID) == int:
            QVSID = str(QVSID)
        availqvs = [os.path.basename(i)[-9:-6] for i in self.qvsnames]
        seqi = availqvs.index(QVSID)
        if seqi < 0:
            print('No seq available')
            return
        CurrentSeriesName = self.qvjroot.find('QVAS_Current_Series_List').find('CurrentSeriesName')
        CurrentSeriesName.text = self.ExamID + 'S' + QVSID + '_' + self.side

    def refreshLoadedSeries(self):
        QVAS_Loaded_Series_List = self.qvjroot.find('QVAS_Loaded_Series_List')
        self.removeLoadedSeries()
        for qvsi in self.qvsnames:
            QVASSeriesFileName = ET.SubElement(QVAS_Loaded_Series_List, 'QVASSeriesFileName')
            QVASSeriesFileName.text = os.path.basename(qvsi)

    def removeLoadedSeries(self):
        QVAS_Loaded_Series_List = self.qvjroot.find('QVAS_Loaded_Series_List')
        # remove existing loaded series
        for qvsi in QVAS_Loaded_Series_List.findall('QVASSeriesFileName'):
            print('Remove existing series', qvsi.text)
            QVAS_Loaded_Series_List.remove(qvsi)

    def setROI(self, xmin, xmax, ymin, ymax):
        ROI = self.qvjroot.find('QVAS_Current_Series_List').find('ROI')
        ROI.set("x1", xmin)
        ROI.set("x2", xmax)
        ROI.set("y1", ymin)
        ROI.set("y2", ymax)

    def writeXML(self):
        self.writeQVJ()
        self.writeQVSall()

    def writeQVJ(self):
        for nodei in list(self.qvjroot):
            nodei.set("xmlns", "")
        xml = ET.tostring(self.qvjroot)
        myfile = open(os.path.join(self.QVJdir, self.ExamID + '.QVJ'), "w")
        myfile.write('<?xml version="1.0" encoding="UTF-8"?>')
        myfile.write(xml.decode("utf-8"))
        myfile.close()
        print('write', os.path.join(self.QVJdir, self.ExamID + '_' + self.side + '.QVJ'))

    def writeQVSall(self):
        availqvs = [os.path.basename(i)[-9:-6] for i in self.qvsnames]
        for qvsi in self.qvsroots:
            for nodei in list(qvsi):
                nodei.set("xmlns", "")
            xml = ET.tostring(qvsi)
            myfile = open(os.path.join(self.QVJdir, self.ExamID + 'S' + availqvs[
                self.qvsroots.index(qvsi)] + '_' + self.side + '.QVS'), "w")
            myfile.write(xml.decode("utf-8"))
            myfile.close()
        # print('write',os.path.join(self.QVJdir,self.ExamID+'S'+availqvs[self.qvsroot.index(qvsi)]+'_'+self.side+'.QVS'))

    def writeQVS(self, QVSID):
        if type(QVSID) == int:
            QVSID = str(QVSID)
        availqvs = [os.path.basename(i)[-9:-6] for i in self.qvsnames]

        if QVSID in availqvs:
            for nodei in list(self.qvsroots[availqvs.index(QVSID)]):
                nodei.set("xmlns", "")
            xml = ET.tostring(self.qvsroots[availqvs.index(QVSID)])
            myfile = open(os.path.join(self.QVJdir, self.ExamID + 'S' + QVSID + '_' + self.side + '.QVS', "w"))
            myfile.write('<?xml version="1.0" encoding="UTF-8"?>')
            myfile.write(xml.decode("utf-8"))
            myfile.close()
        # print('write',os.path.join(self.QVJdir,self.ExamID+'S'+QVSID+'_'+self.side+'.QVS'))
        else:
            print('No QVS', QVSID, 'Available')

    def getBirSlice(self):
        if self.qvjroot.find('QVAS_System_Info').find('BifurcationLocation'):
            bif_slice = int(self.qvjroot.find('QVAS_System_Info').find('BifurcationLocation').find('BifurcationImageIndex').get('ImageIndex'))
            return bif_slice
        else:
            return -1

    def getSliceIQ(self, dicomslicei):
        if self.qvjroot.find('Location_Property') is None:
            raise ValueError('No iq assigned')
        locs = self.qvjroot.find('Location_Property').findall('Location')
        first_slice = int(locs[0].get('Index'))
        bif_slice = self.getBirSlice()
        dcm_offset = bif_slice-(-1-first_slice)
        locid = dicomslicei -dcm_offset
        if locid<0 or locid>=len(locs):
            print(locid,'dcm out of range')
            return 0
        return int(locs[locid].find('IQ').text)

    def listSliceIQ(self):
        if self.qvjroot.find('Location_Property') is None:
            print('No iq assigned')
            return {}
        locs = self.qvjroot.find('Location_Property').findall('Location')
        iq_dict = {}
        for slicei in range(len(locs)):
            index = int(locs[slicei].get('Index'))
            iq_dict[index] = int(locs[slicei].find('IQ').text)
        return iq_dict

    def getContour(self, conttype, dicomslicei, QVSID, roundint=0, scale=1):
        # find qvsroot
        qvsroot = None
        for si in range(len(self.qvsnames)):
            if os.path.basename(self.qvsnames[si])[-9:-6] == QVSID:
                qvsroot = self.qvsroots[si]
                break
        assert qvsroot is not None

        qvasimg = qvsroot.findall('QVAS_Image')
        if dicomslicei - 1 > len(qvasimg):
            print('no slice', dicomslicei)
            return
        assert int(qvasimg[dicomslicei - 1].get('ImageName').split('I')[-1]) == dicomslicei
        conts = qvasimg[dicomslicei - 1].findall('QVAS_Contour')
        tconti = -1
        for conti in range(len(conts)):
            if conts[conti].find('ContourType').text == conttype:
                tconti = conti
                break
        if tconti == -1:
            print('no such contour', conttype)
            return
        pts = conts[tconti].find('Contour_Point').findall('Point')
        contours = []
        for pti in pts:
            contx = float(pti.get('x')) / 512 * self.dcmsz * scale
            conty = float(pti.get('y')) / 512 * self.dcmsz * scale
            if roundint == 1:
                contx = int(round(contx))
                conty = int(round(conty))
            #if current pt is different from last pt, add to contours
            if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
                contours.append([contx, conty])
        return contours

    def setContours(self,QVSID, img_ids, contours, dcmsz, ctx=64, cty=64, scaleres=4, halfpatchsize=256, contour_confs=None, cont_comments=None):
        for tsliceid in range(len(img_ids)):
            tslicei = img_ids[tsliceid]
            contour = contours[tsliceid]
            if contour_confs is not None:
                contour_conf = contour_confs[tsliceid]
            else:
                contour_conf = None
            if cont_comments is not None:
                cont_comment = cont_comments[tsliceid]
            else:
                cont_comment = None
            self.setContour(QVSID,tslicei,contour, dcmsz, ctx, cty, scaleres, halfpatchsize, contour_conf=contour_conf, cont_comment=cont_comment)

    def setContour(self, QVSID, tslicei, contour, dcmsz, ctx=64, cty=64, scaleres=4, halfpatchsize=256, contour_conf=None, cont_comment=None):
        if type(QVSID) == int:
            QVSID = str(QVSID)
        #find qvsroot
        qvsroot = None
        for si in range(len(self.qvsnames)):
            if os.path.basename(self.qvsnames[si])[-9:-6] == QVSID:
                qvsroot = self.qvsroots[si]
                break
        assert qvsroot is not None

        #find qvas_image
        qvasimgs = qvsroot.findall('QVAS_Image')
        fdqvasimg = -1
        for slicei in range(len(qvasimgs)):
            if qvasimgs[slicei].get('ImageName').split('S'+QVSID+'I')[-1] == str(tslicei):
                fdqvasimg = slicei
                break
        if fdqvasimg == -1:
            print('QVAS_IMAGE not found')
            return

        # clear previous contours if there are
        cts = qvasimgs[fdqvasimg].findall('QVAS_Contour')
        for ctsi in cts:
            qvasimgs[fdqvasimg].remove(ctsi)

        #add new contours
        for contype in range(len(contour)):
            if contype == 1:
                ct = "Outer Wall"
                ctcl = '16776960'
            elif contype == 0:
                ct = "Lumen"
                ctcl = '255'

            QVAS_Contour = ET.SubElement(qvasimgs[fdqvasimg], 'QVAS_Contour')
            Contour_Point = ET.SubElement(QVAS_Contour, 'Contour_Point')
            ContourType = ET.SubElement(QVAS_Contour, 'ContourType')
            ContourType.text = ct
            ContourColor = ET.SubElement(QVAS_Contour, 'ContourColor')
            ContourColor.text = ctcl
            ContourOpenStatus = ET.SubElement(QVAS_Contour, 'ContourOpenStatus')
            ContourOpenStatus.text = '1'
            ContourPCConic = ET.SubElement(QVAS_Contour, 'ContourPCConic')
            ContourPCConic.text = '0.5'
            ContourSmooth = ET.SubElement(QVAS_Contour, 'ContourSmooth')
            ContourSmooth.text = '60'
            Snake_Point = ET.SubElement(QVAS_Contour, 'Snake_Point')
            #snake point, fake fill
            for snakei in range(6):
                conti = len(contour[contype])//6*snakei
                Point = ET.SubElement(Snake_Point, 'Point')
                Point.set('x', '%.5f'%((contour[contype][conti][0] / scaleres + ctx - halfpatchsize / scaleres) / dcmsz * 512))
                Point.set('y', '%.5f'%((contour[contype][conti][1] / scaleres + cty - halfpatchsize / scaleres) / dcmsz * 512))

            ContourComments = ET.SubElement(QVAS_Contour, 'ContourComments')
            if cont_comment is not None:
                ContourComments.text = cont_comment
            for coutnodei in contour[contype]:
                Point = ET.SubElement(Contour_Point, 'Point')
                Point.set('x', '%.5f'%((coutnodei[0] / scaleres + ctx - halfpatchsize / scaleres) / dcmsz * 512))
                Point.set('y', '%.5f'%((coutnodei[1] / scaleres + cty - halfpatchsize / scaleres) / dcmsz * 512))

            ContourConf = ET.SubElement(QVAS_Contour, 'ContourConf')
            if contour_conf is not None:
                LumenConsistency = ET.SubElement(ContourConf, 'LumenConsistency')
                LumenConsistency.text = '%.5f'%contour_conf[0]
                WallConsistency = ET.SubElement(ContourConf, 'WallConsistency')
                WallConsistency.text = '%.5f'%contour_conf[1]

    def setBif(self,bif_index,QVSID):
        QVAS_System_Info = self.qvjroot.find('QVAS_System_Info')
        ImageLocationStatus = QVAS_System_Info.find('ImageLocationStatus')
        ImageLocationStatus.text = '2'
        for bi in QVAS_System_Info.findall('BifurcationLocation'):
            QVAS_System_Info.remove(bi)
        BifurcationLocation = ET.SubElement(QVAS_System_Info, 'BifurcationLocation')
        BifurcationImageIndex = ET.SubElement(BifurcationLocation, 'BifurcationImageIndex')
        BifurcationImageIndex.set('ImageIndex',str(bif_index))
        BifurcationImageIndex.set('SeriesName',self.ExamID + 'S' + QVSID + '_' + self.side)
        for nodei in self.qvjroot.findall('QVAS_SeriesForReview_List'):
            self.qvjroot.remove(nodei)
        QVAS_SeriesForReview_List = ET.SubElement(self.qvjroot, 'QVAS_SeriesForReview_List')
        NameOfSFR = ET.SubElement(QVAS_SeriesForReview_List, 'NameOfSFR')
        NameOfSFR.text = self.ExamID + 'S' + QVSID + '_' + self.side

        if type(QVSID) == int:
            QVSID = str(QVSID)
        # find qvsroot
        qvsroot = None
        for si in range(len(self.qvsnames)):
            if os.path.basename(self.qvsnames[si])[-9:-6] == QVSID:
                qvsroot = self.qvsroots[si]
                break
        assert qvsroot is not None

        # find qvas_image
        qvasimgs = qvsroot.findall('QVAS_Image')
        for nodei in self.qvjroot.findall('Location_Property'):
            self.qvjroot.remove(nodei)
        Location_Property = ET.SubElement(self.qvjroot, 'Location_Property')
        Location_Property.set("xmlns", "")
        #last slice always 999
        for relative_si in (np.arange(len(qvasimgs))-bif_index).tolist()+['999']:
            Location = ET.SubElement(Location_Property, 'Location')
            Location.set('Index',str(relative_si))
            #att
            IQ = ET.SubElement(Location, 'IQ')
            IQ.text = '0'
            AHAStatus = ET.SubElement(Location, 'AHAStatus')
            AHAStatus.text = '0.0'
            SurfaceStatus = ET.SubElement(Location, 'SurfaceStatus')
            SurfaceStatus.text = '0'
            FCIntensity = ET.SubElement(Location, 'FCIntensity')
            FCIntensity.text = '-1'
            Intraplaque_Juxta_Hemm = ET.SubElement(Location, 'Intraplaque_Juxta_Hemm')
            Intraplaque_Juxta_Hemm.text = '-1'
            Intraplaque_Juxta_CA = ET.SubElement(Location, 'Intraplaque_Juxta_CA')
            Intraplaque_Juxta_CA.text = '-1'
            Intraplaque_Juxta_LM = ET.SubElement(Location, 'Intraplaque_Juxta_LM')
            Intraplaque_Juxta_LM.text = '-1'
            SurfaceType_Ulcer = ET.SubElement(Location, 'SurfaceType_Ulcer')
            SurfaceType_Ulcer.text = '0'
            SurfaceType_Thrombus = ET.SubElement(Location, 'SurfaceType_Thrombus')
            SurfaceType_Thrombus.text = '0'
            Comments = ET.SubElement(Location, 'Comments')


    def getDicom(self,dicomslicei, QVSID):
        # find qvsroot
        qvsroot = None
        for si in range(len(self.qvsnames)):
            if os.path.basename(self.qvsnames[si])[-9:-6] == QVSID:
                qvsroot = self.qvsroots[si]
                break
        assert qvsroot is not None

        qvasimg = qvsroot.findall('QVAS_Image')
        if dicomslicei - 1 > len(qvasimg):
            print('no slice', dicomslicei)
            return
        assert int(qvasimg[dicomslicei - 1].get('ImageName').split('I')[-1]) == dicomslicei
        dicom_path = self.QVJdir+'/'+qvasimg[dicomslicei - 1].get('ImageName')+'.dcm'
        #print(dicom_path)
        dcm = pydicom.read_file(dicom_path).pixel_array
        return dcm
