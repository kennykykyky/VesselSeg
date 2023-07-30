from ..snakelist import SnakeList
from ..connect.con_utils import mergeSnake
import numpy as np
import scipy

def mergeBranches(icafem,match_gt_all,seg_raw_snakelist,single_only=True):
    match_gt_valid = {}
    for mi in match_gt_all:
        if mi[3]==False:
            continue
        if mi[0] not in match_gt_valid:
            match_gt_valid[mi[0]] = [(mi[1],mi[2])]
        else:
            match_gt_valid[mi[0]].append((mi[1],mi[2]))

    seg_fill_snakelist = SnakeList()
    filled_ids = set()
    for csnake_id in match_gt_valid:
        if len(match_gt_valid[csnake_id])==1:
            sel_ids = [0]
        else:
            if single_only:
                continue
            else:
                sel_ids = [i for i in range(len(match_gt_valid[csnake_id]))]

        for sel_id in sel_ids:
            nei_snake_id = match_gt_valid[csnake_id][sel_id][0]
            reverse_snake = match_gt_valid[csnake_id][sel_id][1]
            print('add',csnake_id,nei_snake_id,reverse_snake)
            filled_ids.add(csnake_id)
            filled_ids.add(nei_snake_id)
            merge_snake_ref = mergeSnake(icafem,seg_raw_snakelist[csnake_id],seg_raw_snakelist[nei_snake_id],reverse_snake,ref_con = True)
            seg_fill_snakelist.addSnake(merge_snake_ref)
    for i in range(seg_raw_snakelist.NSnakes):
        if i in filled_ids:
            continue
        seg_fill_snakelist.addSnake(seg_raw_snakelist[i])
    return seg_fill_snakelist


def mergeSnakeIntMatch(self,ori_int_arr,target_int_arr,sigma = 2):
    ori_int_mean = np.mean(ori_int_arr)
    ori_int_std = max(1, np.std(ori_int_arr))

    norm_distribution = scipy.stats.norm(ori_int_mean, sigma * ori_int_std)

    # if merge, need to compare int fit for both snake
    target_int_mean = np.mean(target_int_arr)

    match_score =  norm_distribution.cdf(target_int_mean)
    if match_score < 0.025 or match_score > 0.975:
        return False
    else:
        return True

def pathMatchInt(self,ori_int_arr,ori_bg_int_arr,interp_int_arr,sigma=1,DEBUG = 0):
    ori_int_arr = excludeAbnormal(ori_int_arr)
    ori_bg_int_arr = excludeAbnormal(ori_bg_int_arr)
    ori_int_mean = np.mean(ori_int_arr)
    ori_int_std = max(1,np.std(ori_int_arr))
    bg_int_mean = np.mean(ori_bg_int_arr)
    bg_int_std = max(1,np.std(ori_bg_int_arr))
    norm_distribution = scipy.stats.norm(ori_int_mean, sigma * ori_int_std)
    bg_norm_distribution = scipy.stats.norm(bg_int_mean, sigma * bg_int_std)
    if DEBUG:
        print('Foreground %.1f+-%.1f, Background %.1f+-%.1f'%(ori_int_mean,ori_int_std,bg_int_mean,bg_int_std))
    '''#test if normal distribution std mixed
    if ori_int_mean-ori_int_std*sigma>bg_int_mean+bg_int_std*sigma:
        #clear foreground background separation, if any signal within background std, invalid
        interp_probs = [norm_distribution.cdf(inti) if inti > bg_int_mean+bg_int_std*sigma else 0 \
                        for inti in interp_int_arr]
        print(interp_probs)
    else:
        # foreground background mixed, int outside foreground std invalid
        interp_probs = [norm_distribution.cdf(inti) \
                        if bg_int_mean - bg_int_std * sigma < inti < bg_int_mean + bg_int_std * sigma else 0 \
                        for inti in  interp_int_arr]
    interp_probs = [i if i < 0.5 else 1 - i for i in interp_probs]'''
    interp_probs = [norm_distribution.pdf(inti)/(bg_norm_distribution.pdf(inti)+norm_distribution.pdf(inti)+1e-6) \
                    for inti in interp_int_arr]

    return interp_probs


def norm_density(x,mean,std):
    ep = 1e-6
    return 1/max(ep,std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(max(ep,2*std**2)))

def excludeAbnormal(arr):
    if len(arr)<=3:
        return arr
    if len(arr)<10:
        arr.remove(max(arr))
        arr.remove(min(arr))
        return arr
    low_thres = np.percentile(arr,10)
    high_thres = np.percentile(arr,90)
    return [x for x in arr if low_thres <= x <= high_thres]
