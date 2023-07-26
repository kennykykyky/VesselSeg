import gudhi as gd
import numpy as np

def betti_error_metric(pred_mask, gt_mask):
    gt_cublical_complex = gd.CubicalComplex(top_dimensional_cells=gt_mask)
    pred_cublical_complex = gd.CubicalComplex(top_dimensional_cells=pred_mask)

    gt_cublical_complex.compute_persistence()
    pred_cublical_complex.compute_persistence()


    gt_bettis = gt_cublical_complex.betti_numbers()
    pred_bettis = pred_cublical_complex.betti_numbers()

    print(gt_bettis)
    print(pred_bettis)



    return 1,2,3





