import numpy as np
from ..snake import Snake

def initSnake(snakei, snakej, npti=5, nptj=5):
    merge_snake_init = Snake()
    leni = min(snakei.NP, npti)
    for i in range(leni):
        merge_snake_init.add(snakei[-leni + i])
    # gap_length = snakei[-1].pos.dist(snakej[0].pos)
    # unit_step = (snakej[0].pos - snakei[-1].pos)/int(round(gap_length))
    # rad_step = (snakej[0].rad - snakei[-1].rad)/int(round(gap_length))
    # for stepi in range(int(round(gap_length))):
    #    merge_snake_init.addSWC(snakei[-1].pos + unit_step * stepi, snakei[-1].rad + rad_step * stepi)
    lenj = min(snakej.NP, nptj)
    for i in range(lenj):
        merge_snake_init.add(snakej[i])

    # print([i.pos for i in merge_snake_init])
    return merge_snake_init  # .resampleSnake(1)


def mergeSnake(icafem, snakei, snakej, reversej=False, ref_con=False):
    merge_snake_init = Snake()
    for i in range(snakei.NP):
        merge_snake_init.add(snakei[i])
    if reversej:
        for i in range(snakej.NP - 1, -1, -1):
            merge_snake_init.add(snakej[i])
    else:
        for i in range(snakej.NP):
            merge_snake_init.add(snakej[i])

    if ref_con:
        ref_pts = 5
        seg_start = max(0, snakei.NP - ref_pts)
        seg_end = min(merge_snake_init.NP, snakei.NP + ref_pts)
        ref_seg = icafem.simpleRefSnake(merge_snake_init.subSnake(seg_start, seg_end))
        merge_ref_snake = Snake(merge_snake_init.snake[:seg_start] + ref_seg.snake + merge_snake_init.snake[seg_end:])
        resample_snake = merge_ref_snake.resampleSnake(1)
        return resample_snake
    else:
        resample_snake = merge_snake_init.resampleSnake(1)
        return resample_snake


def snakeLoss(icafem, snake):
    E_int = snake.posLoss()
    int_arr = icafem.getIntensityAlongSnake(snake, src='s.whole')
    E_img = -np.mean(int_arr)+np.std(int_arr)
    #print('Pos', E_int, 'Img', E_img)
    E_snake = E_int*0 + E_img
    return E_snake


def snakeLossItems(icafem, snake):
    E_int = snake.posLoss()
    int_arr = icafem.getIntensityAlongSnake(snake, src='s.whole')
    E_img = -np.mean(int_arr)
    # print('Pos',E_int,'Img',E_img)
    return [E_int, E_img, int_arr]


def sumSSpos(pos):
    sx = pos[1:] - pos[:-1]
    ssx = sx[1:] - sx[:-1]
    return np.sum(abs(sx))+5*np.sum(abs(ssx))

