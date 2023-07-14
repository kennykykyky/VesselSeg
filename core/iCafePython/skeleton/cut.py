import numpy as np
from iCafePython.snake import Snake
from iCafePython.snakelist import SnakeList

def cutSnakeList(tifimg,seg_raw_snakelist,nstd=3):
    cut_snakelist = SnakeList()
    for snakei in range(seg_raw_snakelist.NSnakes):
        cut_snakelist.extend(cutSnakeIntChange(tifimg, seg_raw_snakelist[snakei], nstd))
    print('before cut',seg_raw_snakelist,'after cut',cut_snakelist)
    return cut_snakelist

def cutSnakeIntChange(tifimg,snake,nstd=3):
    #snake_sig = [icafem.getBoxInt(snake[pti].pos) for pti in range(len(snake))]
    snake_sig = [tifimg[tuple(snake[pti].pos.intlst())] for pti in range(len(snake))]
    cuts = []
    cgroup = []
    for pti in range(len(snake_sig)):
        if len(cgroup)>=5 and (snake_sig[pti] < np.mean(cgroup)-nstd*np.std(cgroup) or snake_sig[pti] > np.mean(cgroup)+nstd*np.std(cgroup)):
            #print('sig break',pti,snake_sig[pti], np.mean(cgroup)-np.std(cgroup), np.mean(cgroup)+np.std(cgroup))
            cuts.append(pti)
            cgroup = []
        elif pti > 1 and pti < len(snake_sig) - 2:
            cur_direct = (snake[pti - 1].pos - snake[pti - 2].pos).norm()
            next_direct = (snake[pti + 2].pos - snake[pti + 1].pos).norm()
            #print('pti', pti, cur_direct, next_direct,cur_direct.prod(next_direct))
            if cur_direct.prod(next_direct) < 0.25:
                cuts.append(pti)
                #print('dir break', pti)
                cgroup = []

        cgroup.append(snake_sig[pti])
    if len(cuts)==0:
        if snake.NP<3:
            snake = snake.resampleSnake(snake)
        return [snake]
    else:
        snakes = []
        cut_pts = [0]+cuts+[len(snake)]
        for c in range(1,len(cut_pts)):
            cut_s = cut_pts[c-1]
            cut_e = cut_pts[c]
            #trim head if needed
            '''for ri in range(2):
                if np.std(snake_sig[cut_s+1:cut_e])<np.std(snake_sig[cut_s:cut_e]):
                    cut_s += 1
                else:
                    break
            #trim tail if needed
            for ri in range(2):
                if np.std(snake_sig[cut_s:cut_e-1])<np.std(snake_sig[cut_s:cut_e]):
                    cut_e -= 1
                else:
                    break'''
            if cut_e-cut_s==1:
                continue
            elif cut_e-cut_s==2:
                snakes.append(Snake(snake[cut_s:cut_e]).resampleSnake(3))
            else:
                snakes.append(Snake(snake[cut_s:cut_e]))
        return snakes
