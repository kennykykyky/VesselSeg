from .con_utils import sumSSpos
from ..snake import Snake
import numpy as np

def snakeFun(x0, *args):
    len_snake = args[0]
    icafem = args[1]

    posx = x0[:len_snake]
    posy = x0[len_snake:2 * len_snake]
    posz = x0[2 * len_snake:]

    E_int = sumSSpos(posx) + sumSSpos(posy) + sumSSpos(posz)
    merge_snake_init = Snake()
    for pti in range(len_snake):
        merge_snake_init.addPt(posx[pti], posy[pti], posz[pti])
    int_arr_o = icafem.getIntensityAlongSnake(merge_snake_init, src='o')
    int_arr_s = icafem.getIntensityAlongSnake(merge_snake_init, src='s.whole')
    E_img = -np.mean(int_arr_o) - np.mean(int_arr_s)
    E_snake = E_int + E_img
    return E_snake


from scipy import optimize
def simpleRefSnake(self,merge_snake_init):
    posx = np.array(merge_snake_init.xlist)
    posy = np.array(merge_snake_init.ylist)
    posz = np.array(merge_snake_init.zlist)
    NP = len(posx)
    res = optimize.minimize(snakeFun, np.concatenate((posx, posy, posz)),
                       args=(NP,self),
                       method='Nelder-Mead',options={'maxiter':1000}
                       )
    merge_snake_ref = Snake()
    for pti in range(NP):
        merge_snake_ref.addPt(res.x[pti],res.x[pti+NP],res.x[pti+NP*2])
    return merge_snake_ref
