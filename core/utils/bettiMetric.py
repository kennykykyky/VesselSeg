import gudhi as gd
import numpy as np
import pdb

def betti_error_metric(pred_mask, gt_mask):
    gt_cublical_complex = gd.CubicalComplex(dimensions = gt_mask.shape, top_dimensional_cells=gt_mask.flatten().astype(int))
    pred_cublical_complex = gd.CubicalComplex(dimensions = pred_mask.shape, top_dimensional_cells=pred_mask.flatten().astype(int))
    

    
    gt_cublical_complex.compute_persistence()
    pred_cublical_complex.compute_persistence()


    gt_bettis = gt_cublical_complex.betti_numbers()
    pred_bettis = pred_cublical_complex.betti_numbers()

    print(gt_bettis)
    print(pred_bettis)


    pdb.set_trace()

    return 1,2,3





