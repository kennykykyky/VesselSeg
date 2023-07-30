import copy
from .con_utils import initSnake, snakeLoss, snakeLossItems
from .snake import simpleRefSnake

def extractVesFeature(icafem,seg_raw_snakelist,mi):
    csnake_id = mi[0]
    nei_snake_id = mi[1]
    reverse_snake = mi[2]

    tsnake = copy.deepcopy(seg_raw_snakelist[nei_snake_id])
    if reverse_snake == 1:
        tsnake.reverseSnake()
    rad_end = seg_raw_snakelist[csnake_id][-1].rad
    rad_head = tsnake[0].rad
    rad_dif = abs(rad_end-rad_head)/(rad_end+rad_head)*2
    merge_snake_init = initSnake(seg_raw_snakelist[csnake_id],tsnake)
    #print(nei_snake_id,'init loss',snakeLoss(icafem,merge_snake_init))
    merge_snake_ref = icafem.simpleRefSnake(merge_snake_init)
    loss_ref_items = snakeLossItems(icafem,merge_snake_ref)
    loss_ref_items.append(rad_dif)
    xlist = merge_snake_ref.xlist
    xlist_rel = [xlist[i]-xlist[0] for i in range(len(xlist))]
    loss_ref_items.append(xlist_rel)
    ylist = merge_snake_ref.ylist
    ylist_rel = [ylist[i] - ylist[0] for i in range(len(ylist))]
    loss_ref_items.append(ylist_rel)
    zlist = merge_snake_ref.zlist
    zlist_rel = [zlist[i] - zlist[0] for i in range(len(zlist))]
    loss_ref_items.append(zlist_rel)
    #loss_ref = snakeLoss(icafem,merge_snake_ref)
    #print(nei_snake_id,'after refine',loss_ref)
    return loss_ref_items

def extractVesFeatures(icafem, seg_raw_snakelist, match_gt):
    feats = []
    y = []
    for mi in match_gt:
        print('\r',mi,'/',len(match_gt),end='')
        loss_ref_items = extractVesFeature(icafem, seg_raw_snakelist,mi)
        feats.append(loss_ref_items)
        y.append(mi[3])
    return feats, y
