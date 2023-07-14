import numpy as np
import copy
import math
from ..utils.img_utils import get_grad_img
import matplotlib.pyplot as plt
import cv2

# rotate patch and predict polar cont along with consistency in rotated patches
def polar_pred_cont_cst(polarpatch,config,polarmodel,gap=10,bioutput=False,usegrad=False):
    DEBUG = 0
    patchheight = config['patchheight']
    width = config['width']
    depth = config['depth']
    channel = config['channel']
    totalheight = config['height']

    # result patch
    # pred_batch_img = np.zeros((totalheight,width,depth,2))
    pred_batch_num = np.zeros((totalheight))
    pred_batch_ctlist = [[] for i in range(totalheight)]

    polarpatchpad = np.zeros((totalheight + patchheight, width, depth, channel))
    polarpatchpad[:totalheight] = polarpatch
    polarpatchpad[totalheight:] = polarpatch[:patchheight]

    test_patch_batch = []

    # collect patches
    for offi in range(0, totalheight, gap):
        polarimgrot = polarpatchpad[offi:offi + patchheight]
        test_patch_batch.append(polarimgrot)

    if bioutput:
        _, predct_batch = polarmodel.predict(np.array(test_patch_batch))
    else:
        predct_batch = polarmodel.predict(np.array(test_patch_batch))

    if usegrad == True:
        polargrad  = get_grad_img(polarpatch[:,:,1,0])

        for bi in range(len(predct_batch)):
            offi = bi * gap
            for slicei in range(patchheight):
                weight = polargrad[(offi + slicei) % totalheight,int(round(predct_batch[bi][slicei][0]*width))]
                #print(bi,(offi + slicei) % totalheight,int(round(predct_batch[bi][slicei][0]*width)),weight)
                pred_batch_ctlist[(offi + slicei) % totalheight].append(predct_batch[bi][slicei]*weight)
                pred_batch_num[(offi + slicei) % totalheight] += weight
    else:
        for bi in range(len(predct_batch)):
            offi = bi * gap
            # print(offi%totalheight,(offi+patchheight-1)%totalheight)
            # pred_batch_img[totalheight-offi:] += predimg_batch[bi][:offi]
            # pred_batch_num[totalheight-offi:] += 1
            # pred_batch_img[:totalheight-offi] += predimg_batch[bi][offi:]
            # pred_batch_num[:totalheight-offi] += 1

            for slicei in range(patchheight):
                pred_batch_ctlist[(offi + slicei) % totalheight].append(predct_batch[bi][slicei])

    # pred_img = np.zeros((totalheight,width,depth,2))
    pred_contour = np.zeros((totalheight, 2))
    pred_contour_sd = np.zeros((totalheight, 2))

    if usegrad == True:
        polargraddsp = copy.copy(polargrad)
        polarpatchdsp = polarpatch[:, :, 1, 0]

    for offi in range(totalheight):
        # pred_img[offi] = pred_batch_img[offi]/pred_batch_num[offi]
        if usegrad == True:
            #print(offi,np.sum(pred_batch_ctlist[offi], axis=0), pred_batch_num[offi])
            pred_contour[offi] = np.sum(pred_batch_ctlist[offi], axis=0)/pred_batch_num[offi] * width
            polargraddsp[offi,int(round(pred_contour[offi][0]))] = np.max(polargraddsp)
            polarpatchdsp[offi,int(round(pred_contour[offi][0]))] = np.max(polarpatchdsp)
        else:
            pred_contour[offi] = np.mean(pred_batch_ctlist[offi], axis=0) * width
        # std dev for uniform distribution in range 0-1 is (1-0)/sqrt(12)
        pred_contour_sd[offi] = np.std(pred_batch_ctlist[offi], axis=0) * np.sqrt(12)

    if DEBUG==1 and usegrad == True:
        plt.imshow(polargraddsp)
        plt.show()
        plt.imshow(polarpatchdsp)
        plt.show()

    # print('batch',len(predimg_batch))
    # print([(i,pred_batch_num[i]) for i in range(len(pred_batch_num))])
    # pyplot.plot([len(i) for i in pred_batch_ctlist])
    # pyplot.show()
    # print([pred_batch_ctlist[0][i][0] for i in range(len(pred_batch_ctlist[0]))])
    # print([pred_batch_ctlist[-1][i][0] for i in range(len(pred_batch_ctlist[-1]))])
    return pred_contour, pred_contour_sd

#from polar contour to cart contour
def toctbd(polarbd,ctx,cty):
    # height, in/out bd, x/y pos
    contour1 = []
    contour2 = []
    for offi in range(polarbd.shape[0]):
        ctheta = 360 / polarbd.shape[0] * offi
        crho = polarbd[offi, 0]
        cx = crho * math.cos(ctheta / 180 * np.pi)
        cy = crho * math.sin(ctheta / 180 * np.pi)
        contour1.append([ctx + cx, cty + cy])
        crho = polarbd[offi, 1]
        cx = crho * math.cos(ctheta / 180 * np.pi)
        cy = crho * math.sin(ctheta / 180 * np.pi)
        contour2.append([ctx + cx, cty + cy])
    contourout = np.array(contour2)
    contourin = np.array(contour1)
    return contourin, contourout

def plotct(sz, contourin, contourout):
    imgmask = np.zeros((sz, sz), dtype=np.uint8)
    intcont = []
    for conti in range(len(contourout)):
        intcont.append([int(round(contourout[conti][0])), int(round(contourout[conti][1]))])
    intcont = np.array(intcont)
    cv2.fillPoly(imgmask, pts=[intcont], color=(1, 1, 1))
    if contourin is not None:
        intcont = []
        for conti in range(len(contourin)):
            intcont.append([int(round(contourin[conti][0])), int(round(contourin[conti][1]))])
        intcont = np.array(intcont)
        cv2.fillPoly(imgmask, pts=[intcont], color=(0, 0, 0))
    # pyplot.imshow(imgmask)
    # pyplot.show()
    return imgmask


def DSC(labelimg,predict_img_thres):
    A = labelimg>0.5*np.max(labelimg)
    B = predict_img_thres>0.5*np.max(predict_img_thres)
    return 2*np.sum(A[A==B])/(np.sum(A)+np.sum(B))
    '''Narr=np.array(labelimg.reshape(-1)>0.5) != np.array(predict_img_thres.reshape(-1))
    FN=np.sum(np.logical_and(Narr, np.array(labelimg.reshape(-1)>0.9)))
    FP=np.sum(Narr)-FN
    Tarr=np.array(labelimg.reshape(-1)>0.5) == np.array(predict_img_thres.reshape(-1))
    TP=np.sum(np.logical_and(Tarr, np.array(labelimg.reshape(-1)>0.9)))
    TN=np.sum(Tarr)-TP
    return 2*TP/(2*TP+FP+FN)'''


def diffmap(A,B):
    diffmap = np.zeros((A.shape[0],A.shape[1],3))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j]==1 and B[i][j]==1:
                diffmap[i,j,2] = 1
            elif A[i][j]==1 and B[i][j]==0:
                diffmap[i,j,1] = 1
            elif A[i][j]==0 and B[i][j]==1:
                diffmap[i,j,0] = 1
    return diffmap
