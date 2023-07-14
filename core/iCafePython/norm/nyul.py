import numpy as np
import xml.etree.ElementTree as ET

percentile = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.998,1]

def createnormLUT(meanper,cper):
    NormLUT=[0]
    for i in range(10):
        for j in range(int(cper[i]),int(cper[i+1])):
            skope = (meanper[i+1]-meanper[i])/(cper[i+1]-cper[i])
            NormLUT.append(int(round(meanper[i]+skope*(j-cper[i]))))
    for j in range(int(cper[10]),int(cper[11])+1):
        NormLUT.append(int(round(meanper[10]+skope*(j-cper[10]))))
    return np.array(NormLUT)


def getPerc(reftifimg):
    refperc = np.zeros((len(percentile)))
    for peri in range(len(percentile)):
        refperc[peri] = np.percentile(reftifimg[reftifimg > 0], percentile[peri] * 100)

    return refperc

def nyulFromNormImg(newtifimg, reftifimg):
    refperc = getPerc(reftifimg)
    newperc = getPerc(newtifimg)
    cNormLUT = createnormLUT(refperc, newperc)
    normimg = cNormLUT[newtifimg]
    return normimg

def nyulFromNormPerc(newtifimg, refperc):
    newperc = getPerc(newtifimg)
    #print(refperc, newperc)
    cNormLUT = createnormLUT(refperc, newperc)
    #print(cNormLUT)
    normimg = cNormLUT[newtifimg]
    return normimg

def readMeanHist(db_xml_path):
    root = ET.parse(db_xml_path).getroot()
    perc_node = root.find('Percentile')
    refperc = [0]
    for p in percentile[1:-2]:
        refperc.append(int(perc_node.find('per%d'%(p*100)).text))
    refperc.append(int(perc_node.find('per998').text))
    #100
    refperc.append(refperc[-1]+1)
    return refperc
