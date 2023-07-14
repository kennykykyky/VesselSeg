import numpy as np


def DSC(pred, label, thres=0.5):
    pred = pred > thres * np.max(pred)
    comm = np.sum(np.bitwise_and(label, pred))
    label_sum = np.sum(label)
    pred_sum = np.sum(pred)
    return 2 * comm / (label_sum + pred_sum)
