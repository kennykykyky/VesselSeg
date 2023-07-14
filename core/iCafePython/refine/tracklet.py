import os
from ..BB import BB

def collectbb(bbfolder,class_num):
    if len(os.listdir(bbfolder))==0:
        print('no bb in folder',bbfolder)
        return
    bbr = [[] for i in range(len(os.listdir(bbfolder)))]
    for slicepathi in range(len(os.listdir(bbfolder))):
        bbfilelist = os.listdir(bbfolder)
        slicei = int(bbfilelist[slicepathi][:-4].split('_')[-1])
        slicename = os.path.join(bbfolder,bbfilelist[slicepathi])
        with open(slicename, 'r') as file:
            bbnum = int(file.readline()[:-1])
            if bbnum==0:
                continue
            cbbs = []
            for bbi in range(bbnum):
                cbblist = [float(i) for i in file.readline()[:-1].split(' ')]
                cbblist[-1] = int(cbblist[-1])
                cbbs.append(BB.fromminmaxlistclabel(cbblist,class_num))
            bbr[slicei].extend(cbbs)
    return bbr

