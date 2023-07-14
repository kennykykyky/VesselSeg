from ..snakelist import SnakeList
from ..point3d import Point3D
from ..swcnode import SWCNode
from ..snake import Snake
import numpy as np

def paint_ball(img_fill, ct, rad, target=0):
    ct_int = ct.intlst()
    rad_int = int(round(rad))
    posz = min(img_fill.shape[0] - 1, ct_int[2])
    posx = min(img_fill.shape[1] - 1, ct_int[0])
    posy = min(img_fill.shape[2] - 1, ct_int[1])
    img_fill[posz, posx, posy] = 0
    for ofx in range(-rad_int, rad_int):
        for ofy in range(-rad_int, rad_int):
            for ofz in range(-rad_int, rad_int):
                cpos = Point3D([ct_int[0] + ofx, ct_int[1] + ofy, ct_int[2] + ofz])
                cpos.boundList([img_fill.shape[1], img_fill.shape[2], img_fill.shape[0]])
                cdist = cpos.dist(ct)
                if cdist > rad:
                    continue
                posz = min(img_fill.shape[0] - 1, cpos.intlst()[2])
                posx = min(img_fill.shape[1] - 1, cpos.intlst()[0])
                posy = min(img_fill.shape[2] - 1, cpos.intlst()[1])
                img_fill[posz, posx, posy] = target


def createLabelMap(seg_ves_snakelist, shape):
    label_img = np.ones(shape, dtype=np.int16) * (-1)
    for snakeid in range(seg_ves_snakelist.NSnakes):
        if snakeid % 5 == 0:
            print('\rpainting snake', snakeid, '/',seg_ves_snakelist.NSnakes, end='')
        for pti in range(seg_ves_snakelist._snakelist[snakeid].NP):
            pos = seg_ves_snakelist._snakelist[snakeid][pti].pos
            rad = seg_ves_snakelist._snakelist[snakeid][pti].rad
            # paint within radius at pos position
            paint_ball(label_img, pos, rad, 0)
    return label_img


from .build_vessel_tree import search_first_node, search_one_direction


def extend_seg_end(infer_model, re_spacing_img, traced_img, chead, cdir_mv):
    # chead: current segment ending position
    # cdir_mv: move direction based on first two pts on the original segment
    prob_records = [0, 0, 0]
    max_points = 500
    prob_thr = 0.85
    res = search_first_node(infer_model, re_spacing_img, start=chead, prob_records=prob_records)
    add_seg = Snake()
    if res is None:
        print('not valid patch')
    else:
        direction, prob_records, curr_r = res
        if prob_records[-1] > prob_thr:
            print('\rfirst prob', prob_records[-1], 'higher than termination prob_thr', prob_thr,end='')
        else:
            forward_matchness = Point3D(direction['forward_vector']).norm() * cdir_mv
            backward_matchness = Point3D(direction['backward_vector']).norm() * cdir_mv
            if forward_matchness > backward_matchness:
                sel_start_pos = direction['forward']
                sel_direction = direction['forward_vector']
            else:
                sel_start_pos = direction['backward']
                sel_direction = direction['backward_vector']

            point_list = []
            r_list = []

            find_node = search_one_direction(infer_model, traced_img, re_spacing_img, start=sel_start_pos,
                                             move_direction=sel_direction,
                                             prob_records=prob_records,
                                             r_list=r_list, point_list=point_list, step_ratio=0.5)
            for pti in range(len(point_list)):
                add_seg.addSWC(Point3D(point_list[pti]), r_list[pti])
    return add_seg

def extSnake(seg_ves_snakelist,infer_model,re_spacing_img,DEBUG=0):
    # traced_img = seg_ves_snakelist.labelMap(icafem.shape)
    traced_img = createLabelMap(seg_ves_snakelist, re_spacing_img.shape)

    for snakei in range(seg_ves_snakelist.NSnakes):
        seg_ves_snake = seg_ves_snakelist[snakei]
        chead = seg_ves_snake[0].pos.lst()
        move_dir_head = (seg_ves_snake[0].pos - seg_ves_snake[1].pos).norm()

        ctail = seg_ves_snake[-1].pos.lst()
        move_dir_tail = (seg_ves_snake[-1].pos - seg_ves_snake[-2].pos).norm()

        # add seg ahead of head of snake
        # set starting point as -1 to allow stretching
        paint_ball(traced_img, seg_ves_snake[0].pos, seg_ves_snake[0].rad, -1)
        add_seg = extend_seg_end(infer_model, re_spacing_img, traced_img, chead, move_dir_head)
        # restore
        paint_ball(traced_img, seg_ves_snake[0].pos, seg_ves_snake[0].rad, 0)
        if add_seg.NP > 0:
            if DEBUG:
                print(snakei, 'head prepend', add_seg)
            seg_ves_snake.mergeSnake(add_seg, reverse=False, append=False)
        else:
            if DEBUG:
                print(snakei, 'head no extend')
        # add seg after tail of snake
        paint_ball(traced_img, seg_ves_snake[-1].pos, seg_ves_snake[-1].rad, -1)
        add_seg = extend_seg_end(infer_model, re_spacing_img, traced_img, ctail, move_dir_tail)
        paint_ball(traced_img, seg_ves_snake[-1].pos, seg_ves_snake[-1].rad, 0)
        if add_seg.NP > 0:
            if DEBUG:
                print(snakei, 'tail add', add_seg)
            seg_ves_snake.mergeSnake(add_seg, reverse=False, append=True)
        else:
            if DEBUG:
                print(snakei, 'tail no extend')
    return seg_ves_snakelist
