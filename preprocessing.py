import numpy as np
import os
import argparse
import pdb
import shutil
from PIL import Image
from core.iCafePython import iCafe

'''
This scirpt intends to clean up the data from iCafe 'results' folder and return the cleaned data 
for lumen segmentation models

This script will use iCafe class and other AICafe functions (may switch the virtual environment)

The cleaned data includes the original 3D/4D img (*.npy) and the corresponding 
binary segmentation mask (distance transform mask) (*.dnpy)

The output data will be stored under data/'dataset_name'/
'''

def parse_option():
    parser = argparse.ArgumentParser('argument for generating lumen segmentation mask')

    parser.add_argument('--data_dir',
                        type=str,
                        default='P:/iCafe/result/iSNAP')
    # parser.add_argument('--vessel_id_file', type=str, default = None)
    parser.add_argument('--save_dir',
                        type=str,
                        default='D:/Kaiyu/vesselseg/data/iSNAP')

    FLAGS = parser.parse_args()

    return FLAGS

def main():

    FLAGS = parse_option()

    # First, get the data list from iCafe results folder
    caselist = [f.path for f in os.scandir(FLAGS.data_dir) if f.is_dir() and len(f.name) < 14]

    # Second, read the original 3D/4D img and save it in .npy file under data/
    caselist_empty = []
    for case in caselist:
        casename = 'TH_' + case.split('\\')[-1]
        np_file = os.path.join(case, casename + '.npy')
        if os.path.exists(np_file):
            tar_dir = FLAGS.save_dir + '/' + case.split('\\')[-1] + '/'
            if not os.path.exists(tar_dir):
                os.mkdir(tar_dir)
            if os.path.exists(tar_dir + casename + '.npy'): continue
            shutil.copyfile(np_file, tar_dir + casename + '.npy')
        else:
            tif_file = os.path.join(case, casename + '.tif')
            if not os.path.exists(tif_file):
                print('No .npy file or .tif file found for {}'.format(casename))
                caselist_empty.append(case)
                continue
            tif_npy = np.array(Image.open(tif_file))
            np.save(tif_npy, FLAGS.save_dir + '/' + case.split('\\') + '/' + casename + '.tif')

    # Third, generate the segmentation mask based on the snakelist (.swc) field and save '*d.npy' under iCafe results folder
    caselist = [ele for ele in caselist if ele not in caselist_empty]
    for case in caselist:
        icafem = iCafe(case)
        icafem.loadImg('o')
        if not icafem.existPath('seg_ves') and not icafem.existPath('raw_ves'): continue
        if icafem.existPath('d.npy'):
            continue
        snakelist = icafem.snakelist
        icafem.paintDistTransform(snakelist)

    # Finally, transform the '*d.npy' file (x*y*z*C), where C includes the position and the radius of the point in snakelist,
    # into '*seg.npy' which is just the binary segmentation mask
    
    # Here just copy the '*d.npy' file with (x, y, z, radius) information
    for case in caselist:
        casename = 'TH_' + case.split('\\')[-1]
        tar_dir = FLAGS.save_dir + '/' + case.split('\\')[-1] + '/'
        d_file = os.path.join(case, casename + 'd.npy')
        if os.path.exists(tar_dir + casename + 'd.npy'): continue
        if os.path.exists(os.path.join(case, casename + 'd.npy')):
            shutil.copyfile(d_file, tar_dir + casename + 'd.npy')
            
    # Here we just copy the binary segmentation mask
    for case in caselist:
        casename = 'TH_' + case.split('\\')[-1]
        d_file = os.path.join(case, casename + 'd.npy')
        seg = np.load(d_file)
        seg = (seg[..., 0] > 0).astype(np.float32)
        tar_dir = FLAGS.save_dir + '/' + case.split('\\')[-1] + '/'
        if os.path.exists(tar_dir + casename + 'seg.npy'): continue
        else:
            np.save(tar_dir + casename + 'seg.npy', seg)
                
if __name__ == '__main__':
    main()