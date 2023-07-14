import os
import pickle

def getArea(m_fpoints,scale=1):
	dArea = 0
	nSize = len(m_fpoints)
	if nSize<=2:
		print("Not enough points to get contour area.")
		return 0
	for idx in range(nSize-1):
		dArea += m_fpoints[idx][0]*scale * m_fpoints[idx+1][1]*scale
	dArea += m_fpoints[nSize-1][0]*scale * m_fpoints[0][1]*scale

	for idx in range(nSize-1):
		dArea -= m_fpoints[idx][1]*scale * m_fpoints[idx+1][0]*scale

	dArea -= m_fpoints[nSize-1][1]*scale * m_fpoints[0][0]*scale
	dArea = dArea /2.0

	dArea = abs(dArea)
	return dArea


def writeContour(lumencontourname,wallcontourname,lumencontour,outerwallcontour,scale=1):
    contourfile = open(lumencontourname, "w")
    for contindx in range(len(lumencontour)):
        coutnodei = lumencontour[contindx]
        contourfile.write("%.2f %.2f\n"%(coutnodei[0]*scale,coutnodei[1]*scale))
    #repeat first point
    contourfile.write("%.2f %.2f\n"%(lumencontour[0][0]*scale,lumencontour[0][1]*scale))
    contourfile.close()

    contourfile = open(wallcontourname, "w")
    for contindx in range(len(outerwallcontour)):
        coutnodei = outerwallcontour[contindx]
        contourfile.write("%.2f %.2f\n"%(coutnodei[0]*scale,coutnodei[1]*scale))
    #repeat first point
    contourfile.write("%.2f %.2f\n"%(outerwallcontour[0][0]*scale,outerwallcontour[0][1]*scale))
    contourfile.close()

#wtd_path is the path to the vw measurement exe "D:\iCafe\wt\getwtd.exe"
def measureVW(lumen_contour,wall_contour,wtd_exe,wall_contour_name=None,lumen_contour_name=None,stat_name=None):
    #if no wall contour filename defined, save contours in the same folder as getwtd.exe
    if wall_contour_name is None:
        vwd_folder = os.path.abspath(os.path.join(wtd_exe, os.pardir))
        wall_contour_name = vwd_folder+'/OuterWall.txt'
    if lumen_contour_name is None:
        vwd_folder = os.path.abspath(os.path.join(wtd_exe, os.pardir))
        lumen_contour_name = vwd_folder+'/Lumen.txt'

    #write the contours to txt files
    writeContour(wall_contour_name,lumen_contour_name,lumen_contour,wall_contour)
    #then run the c++ code to read txt tile and measure
    vw_cal_command = wtd_exe + wall_contour_name + ' ' + lumen_contour_name
    #print(vw_cal_command)
    #print('calculate vessel wall stats using dll',vwcalcommand)
    statouput = os.popen(vw_cal_command).readlines()
    #save stat to txt
    if stat_name is not None:
        #wall_contour_name.replace('OuterWall','VWStat')
        with open(stat_name, 'w') as statfile:
            statfile.write(''.join(statouput))
        statfile.close()

    #add new result
    if len(statouput)<5:
        print('ERR feat extraction')

    SCALE = 1
    maxThickness = float(statouput[1].split(':')[1][:-1]) /SCALE
    minThickness = float(statouput[2].split(':')[1][:-1]) /SCALE
    avgThickness = float(statouput[3].split(':')[1][:-1]) /SCALE
    stdThickness = float(statouput[4].split(':')[1][:-1]) /SCALE

    arealumen = getArea(lumen_contour)/SCALE/SCALE
    areawall = getArea(wall_contour)/SCALE/SCALE
    return maxThickness,minThickness,avgThickness,stdThickness,arealumen,areawall


def measureVWs(art_contours,wtd_exe):
    vw_measure = []
    for pti in range(len(art_contours)):
        print('\rVW measure',pti,'/',len(art_contours),end='')
        lumen_contour = art_contours[pti][0]
        wall_contour = art_contours[pti][1]
        maxThickness,minThickness,avgThickness,stdThickness,arealumen,areawall = measureVW(lumen_contour,wall_contour,wtd_exe)
        vw_measure.append([pti,maxThickness,minThickness,avgThickness,stdThickness,arealumen,areawall])
    return vw_measure

def measureVWArt(self,art_contours,wtd_exe=None):
    if wtd_exe is None:
        wtd_exe = self.icafe_base_name + '/wt/getwtd.exe '
    return measureVWs(art_contours,wtd_exe)

def saveMeasureResult(self,vw_result):
    vw_result_file = self.path + '/vw_measure.pickle'
    with open(vw_result_file, 'wb') as fp:
        pickle.dump(vw_result, fp)

def loadMeasureResult(self):
    vw_result_file = self.path + '/vw_measure.pickle'
    if not os.path.exists(vw_result_file):
        return None
    with open(vw_result_file, 'rb') as fp:
        vw_result = pickle.load(fp)
    return vw_result
