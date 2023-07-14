import numpy as np
import torch
import copy
from .build_vessel_utils import  data_preprocess, prob_terminates, get_shell, get_angle
prob_thr = 0.85
max_points = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer(infer_model, re_spacing_img, start: list):
    """
    :param start: Initial point
    :return: Moving position, the index of maximum confidence direction, Current termination probability
    """
    max_z = re_spacing_img.shape[0]
    max_y = re_spacing_img.shape[1]
    max_x = re_spacing_img.shape[2]

    cut_size = 9
    '''spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    center_x_pixel = get_spacing_res2(start[0], spacing_x, resize_factor[1])
    center_y_pixel = get_spacing_res2(start[1], spacing_y, resize_factor[2])
    center_z_pixel = get_spacing_res2(start[2], spacing_z, resize_factor[0])'''

    center_x_pixel = int(round(start[0]))
    center_y_pixel = int(round(start[1]))
    center_z_pixel = int(round(start[2]))


    left_x = center_x_pixel - cut_size
    right_x = center_x_pixel + cut_size
    left_y = center_y_pixel - cut_size
    right_y = center_y_pixel + cut_size
    left_z = center_z_pixel - cut_size
    right_z = center_z_pixel + cut_size

    new_patch = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))

    if not (
            left_x < 0 or right_x < 0 or left_y < 0 or right_y < 0 or \
            left_x >= max_x or right_x >= max_x or left_y >= max_y or \
            right_y >= max_y):
        for ind in range(left_z, right_z + 1):
            if ind<0 or ind>=max_z:
                continue
            src_temp = re_spacing_img[ind].copy()
            new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
        input_data = data_preprocess(new_patch)

        inputs = input_data.to(device)
        outputs = infer_model(inputs.float())

        outputs = outputs.view((len(input_data), max_points + 1))
        outputs_1 = outputs[:, :len(outputs[0]) - 1]
        outputs_2 = outputs[:, -1]

        outputs_1 = torch.nn.functional.softmax(outputs_1, 1)
        indexs = np.argsort(outputs_1.cpu().detach().numpy()[0])[::-1]
        curr_prob = prob_terminates(outputs_1, max_points).cpu().detach().numpy()[0]
        curr_r = outputs_2.cpu().detach().numpy()[0]
        sx, sy, sz = get_shell(max_points, curr_r)
        return [sx, sy, sz], indexs, curr_r, curr_prob
    else:
        print('out of bound',left_x,right_x,left_y,right_y ,left_z,right_z)
        return None


def search_first_node(infer_model, re_spacing_img, start: list, prob_records: list):
    """
    :param start: Initial point
    :return: Next direction vector, Probability record, Current radius
    """
    res = infer(infer_model, re_spacing_img, start=start)
    if res is None:
        return
    s_all, indexs, curr_r, curr_prob = res
    start_x, start_y, start_z = start
    prob_records.pop(0)
    prob_records.append(curr_prob)
    sx, sy, sz = s_all
    forward_x = sx[indexs[0]] + start_x
    forward_y = sy[indexs[0]] + start_y
    forward_z = sz[indexs[0]] + start_z
    forward_move_direction_x = sx[indexs[0]]
    forward_move_direction_y = sy[indexs[0]]
    forward_move_direction_z = sz[indexs[0]]
    for i in range(1, len(indexs)):
        curr_angle = get_angle(np.array([sx[indexs[i]], sy[indexs[i]], sz[indexs[i]]]),
                               np.array([forward_move_direction_x, forward_move_direction_y, forward_move_direction_z]))
        # To determine two initial opposing directions of the tracker, two local maxima d0 and d0′ separated by an angle ≥ 90°
        if curr_angle >= 90:
            backward_move_direction_x = copy.deepcopy(sx[indexs[i]])
            backward_move_direction_y = copy.deepcopy(sy[indexs[i]])
            backward_move_direction_z = copy.deepcopy(sz[indexs[i]])
            break
    backward_x = backward_move_direction_x + start_x
    backward_y = backward_move_direction_y + start_y
    backward_z = backward_move_direction_z + start_z
    direction = {}
    direction["forward"] = [forward_x, forward_y, forward_z]
    direction["forward_vector"] = [forward_move_direction_x, forward_move_direction_y, forward_move_direction_z]
    direction["backward"] = [backward_x, backward_y, backward_z]
    direction["backward_vector"] = [backward_move_direction_x, backward_move_direction_y, backward_move_direction_z]
    return direction, prob_records, curr_r


def move(start: list, shell_arr: list, indexs: list, move_direction: list, step_ratio=1):
    """
    Moving ball
    :param start: start point
    :param shell_arr: shell arr
    :param indexs: index of next direction
    :param move_direction: last move direction
    :param curr_r: radius
    :return: direction vector, move to next point
    """
    start_x, start_y, start_z = start
    sx, sy, sz = shell_arr
    move_direction_x, move_direction_y, move_direction_z = move_direction
    for i in range(len(indexs)):
        curr_angle = get_angle(np.array([sx[indexs[i]], sy[indexs[i]], sz[indexs[i]]]),
                               np.array([move_direction_x, move_direction_y, move_direction_z]))
        # Only directions with an angle ≤ 60°to the previously followed direction are considered.

        if curr_angle <= 60:
            new_x = sx[indexs[i]]*step_ratio + start_x
            new_y = sy[indexs[i]]*step_ratio + start_y
            new_z = sz[indexs[i]]*step_ratio + start_z
            move_direction_x = sx[indexs[i]]
            move_direction_y = sy[indexs[i]]
            move_direction_z = sz[indexs[i]]
            break

    return [move_direction_x, move_direction_y, move_direction_z], [new_x, new_y, new_z]

def paintBall(traced_img, cpos, rad, paint_val):
    xmin = int(np.floor(max(0, cpos[0] - rad)))
    xmax = int(np.ceil(min(traced_img.shape[1] - 1, cpos[0] + rad)))
    ymin = int(np.floor(max(0, cpos[1] - rad)))
    ymax = int(np.ceil(min(traced_img.shape[2] - 1, cpos[1] + rad)))
    zmin = int(np.floor(max(0, cpos[2] - rad)))
    zmax = int(np.ceil(min(traced_img.shape[0] - 1, cpos[2] + rad)))

    for xi in range(xmin, xmax + 1):
        for yi in range(ymin, ymax + 1):
            for zi in range(zmin, zmax + 1):
                if np.sqrt((xi-cpos[0])**2+(yi-cpos[1])**2+(zi-cpos[2])**2) <= rad:
                    if traced_img[zi, xi, yi]==-1:
                        traced_img[zi, xi, yi] = paint_val
    return traced_img

def search_one_direction(infer_model, traced_img,re_spacing_img, start: list, move_direction: list,
                         prob_records: list, point_list: list,
                         r_list: list, root=None, find_node=None,step_ratio=1):
    """
    :param start: start point
    :param move_direction: last move direction
    :param prob_records: record of termination probability
    :param point_list:
    :param r_list: radius arr
    :return:
    """
    find_node_initial = None
    prob_mean = sum(prob_records) / len(prob_records)
    MAXIT = 500
    it = 0
    next_point = start
    while prob_mean <= prob_thr and find_node_initial is None and it<MAXIT:
        it += 1
        if it%10==0:
            print('\r',it,'/',MAXIT,'iterations',end='')
        result = infer(infer_model, re_spacing_img, start=next_point)
        if result is not None:
            #print(it,next_point,traced_img[min(traced_img.shape[0]-1,int(round(next_point[2]))),
            #              min(traced_img.shape[1]-1,int(round(next_point[0]))),
            #              min(traced_img.shape[2]-1,int(round(next_point[1])))] )
            if traced_img[min(traced_img.shape[0]-1,int(round(next_point[2]))),
                          min(traced_img.shape[1]-1,int(round(next_point[0]))),
                          min(traced_img.shape[2]-1,int(round(next_point[1])))] in np.arange(0,it-5).tolist()+[0]:
                break
            if next_point[2]<0 or next_point[2]>traced_img.shape[0]-1:
                break
            shell_arr, indexs, curr_r, curr_prob = result
            r_list.append(curr_r)
            point_list.append(next_point)
            prob_records.pop(0)
            prob_records.append(curr_prob)
            prob_mean = sum(prob_records) / len(prob_records)
            move_direction, next_point = move(start=next_point, shell_arr=shell_arr, indexs=indexs,
                                         move_direction=move_direction,step_ratio=step_ratio)
            #print('next_point',next_point)
            paintBall(traced_img, next_point, curr_r, it)
            #if find_node is None:
            #    find_node_initial = search_tree(root, next_point)
        else:
            break
    #set as previous trace
    traced_img[traced_img>0] = 0
    #print('px',np.sum(traced_img==0))
    return find_node_initial
