from medpy.metric.binary import assd
import numpy as np

def ahd_metric(pred_img, gt):
    # no stenosis for now; copied code
    pos_s = -1

    ahd = assd(pred_img, gt)

    # it's normalized using some excel sheet with values
    # ahd_score = max((nc - ahd) / nc, 0)

    return ahd



def DSC(pred_img, label_img):
    A = label_img > 0.5 * np.max(label_img)
    B = pred_img > 0.5 * np.max(pred_img)
    return 2*np.sum(A[A==B])/(np.sum(A)+np.sum(B))


def _compute_norm_factor(img, mask, spacing, pos_s):
    # compute the norm facter of AHD scores
    # , using AHD scores of segmentation by simple threshold
    ahd = []
    ahd_s = []
    for th in [0.2, 0.3, 0.4]:
        pred_img = img > img.max() * th
        ahd.append(assd(pred_img, mask, voxelspacing=spacing))

    if pos_s != -1:
        mask_s = mask[pos_s[0]:pos_s[1], pos_s[2]:pos_s[3], pos_s[4]:pos_s[5]]
        pred_img_s = pred_img[pos_s[0]:pos_s[1], pos_s[2]:pos_s[3], pos_s[4]:pos_s[5]]
        if pred_img_s.max() == False:
            pred_img_s[int(pred_img_s.shape[0]/2), int(pred_img_s.shape[1]/2), int(pred_img_s.shape[2]/2)] = 1
        ahd_s.append(assd(pred_img_s, mask_s, voxelspacing=spacing))
    else:
        ahd_s.append(-1)

    nf = np.mean(ahd)
    nf_s = np.mean(ahd_s) if ahd_s[0] != -1 else -1
    return nf, nf_s