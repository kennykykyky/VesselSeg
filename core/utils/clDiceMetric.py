from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
import pdb

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """

    tprec = None
    tsens = None
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    elif len(v_p.shape)==4: # during training
        # use for loop to calculate the cldice metric and then calculate the mean
        tprec_sum = 0
        tsens_sum = 0
        for i in range(v_p.shape[0]):
            tprec_sum = cl_score(v_p[i,...],skeletonize(v_l[i,...]))
            tsens_sum = cl_score(v_l[i,...],skeletonize(v_p[i,...]))
        tprec = tprec_sum/v_p.shape[0]
        tsens = tsens_sum/v_p.shape[0]
    return 2*tprec*tsens/(tprec+tsens)