import numpy as np
# from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import eval_linear
from scipy.optimize import minimize
from ..point3d import Point3D
import copy

def opfun(bd, *args):
    DEBUG = 1
    y1 = copy.copy(args[0])
    y2 = args[1]
    steps = args[2]
    lenbd = args[3]
    intmprimg = args[4]
    snake= args[5]
    mprimg = args[6]

    # print('b',icafem.snakelist[snakeid][2].pos)

    numnodes = len(snake)
    x1 = bd[:lenbd]
    x2 = bd[lenbd:lenbd * 2]
    posx = bd[lenbd * 2:lenbd * 2 + numnodes]
    posy = bd[lenbd * 2 + numnodes:lenbd * 2 + numnodes * 2]
    posz = bd[lenbd * 2 + numnodes * 2:lenbd * 2 + numnodes * 3]
    # print(bd[:5],bd[-5:])
    # difmprimg = np.gradient(gaussian(mprimg, 3), axis=1)
    difmprimg = np.gradient(gaussian_filter(mprimg, 3), axis=1)

    if DEBUG and steps[0] % 1000 == 0:
        plt.imshow(mprimg)
        plt.colorbar()
        # plt.colorbar()
        lwidth = 2
        plt.plot(x1, y1, '--r', lw=lwidth)
        # plt.plot(snake1[:, 0], snake1[:, 1], '-b', lw=lwidth)
        plt.plot(x2, y2, '--r', lw=lwidth)
        # plt.plot(snake2[:, 0], snake2[:, 1], '-b', lw=lwidth)
        plt.show()

    vs1 = x1[1:] - x1[:-1]
    vss1 = vs1[1:] - vs1[:-1]
    vs2 = x2[1:] - x2[:-1]
    vss2 = vs2[1:] - vs2[:-1]
    smoothloss = (np.mean(abs(vs1)) + np.mean(abs(vs2)) + 5 * np.mean(abs(vss1)) + 5 * np.mean(abs(vss2))) / 1
    intensityloss = -np.mean([eval_linear(intmprimg, mprimg, np.array([y1[i], x1[i]])) for i in range(len(y1))]) \
                    - np.mean([eval_linear(intmprimg, mprimg, np.array([y2[i], x2[i]])) for i in range(len(y2))])
    gradientloss = 40 * np.mean([eval_linear(intmprimg, difmprimg, np.array([y1[i], x1[i]])) for i in range(len(y1))]) \
                   - 40 * np.mean([eval_linear(intmprimg, difmprimg, np.array([y2[i], x2[i]])) for i in range(len(y2))])
    distlosses = []
    for i in range(len(y1)):
        if x1[i] < x2[i]:
            distlosses.append(1000)
        elif x1[i] - x2[i] < 10:
            # distlosses.append(-np.log(kernel(x1[i]/2-x2[i]/2))-minloss)
            distlosses.append(0)
        else:
            distlosses.append(np.exp((x1[i] - x2[i]) / 10) - 1)
    distloss = np.mean(distlosses) / 10
    distchange = (x1[1:] - x2[1:]) - (x1[:-1] - x2[:-1])
    distchangeloss = np.mean(abs(distchange)) / 1

    posdif = []
    for i in range(1, len(posx)):
        posdif.append(Point3D(posx[i], posy[i], posz[i]).dist(Point3D(posx[i - 1], posy[i - 1], posz[i - 1])))
    posdif = np.array(posdif)
    ptdistloss = np.mean(abs(posdif[1:] - posdif[:-1]))

    centerpos = (x1 + x2) / 2
    centerx = 50
    losscenter = np.mean(abs(centerpos - centerx))

    if DEBUG:
        if steps[0] % 100 == 0:
            print(steps[0], smoothloss, intensityloss, gradientloss, distloss, distchangeloss, ptdistloss, losscenter)
            #print(snake[2].pos, posx[2], posy[2], posz[2])
    for j in range(len(posx)):
        snake[j].pos = Point3D(posx[j], posy[j], posz[j])

    steps[0] += 1
    return smoothloss + intensityloss + gradientloss + distloss + distchangeloss + ptdistloss + losscenter

def refSnakeList(self, snakelist, dsp=0):
    for snakei in range(snakelist.NSnakes):
        print('Ref snake',snakei)
        self.refSnake(snakelist[snakei],dsp)

def refSnake(self, snake, dsp=0):
    if snake.NP < 5:
        return
    mprimg = self.mpr(snake, 's', rot=0)

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for ptidi in range(snake.NP):
        if ptidi % 3 == 0 or ptidi == snake.NP - 1:
            x1.append(mprimg.shape[1] // 2 + snake[ptidi].rad)
            y1.append(snake.getAccLen(ptidi))
            x2.append(mprimg.shape[1] // 2 - snake[ptidi].rad)
            y2.append(snake.getAccLen(ptidi))
    bd1 = np.array([x1, y1]).T
    bd2 = np.array([x2, y2]).T

    mprimg = mprimg - np.min(mprimg)
    mprimg = mprimg / np.max(mprimg)

    #difmprimg = np.gradient(gaussian(mprimg, 3), axis=1)
    # plt.imshow(difmprimg)
    # plt.colorbar()
    # plt.show()
    posx = [snake[i].pos.x for i in range(snake.NP)]
    posy = [snake[i].pos.y for i in range(snake.NP)]
    posz = [snake[i].pos.z for i in range(snake.NP)]

    intmprimg = UCGrid((0, mprimg.shape[0] - 1, mprimg.shape[0]),
                       (0, mprimg.shape[1] - 1, mprimg.shape[1]))
    steps = [0]
    res = minimize(opfun, np.concatenate((bd1[:, 0], bd2[:, 0], posx, posy, posz)),
                   args=(bd1[:, 1], bd2[:, 1], steps, len(bd1), intmprimg, snake, mprimg),
                   method='Nelder-Mead', options={'maxiter':5000}
                   )

    snake1 = np.array([res.x[:len(bd1)], bd1[:, 1]]).T
    snake2 = np.array([res.x[len(bd1):2 * len(bd1)], bd1[:, 1]]).T

    if dsp:
        plt.figure(figsize=(mprimg.shape[0] / 10, mprimg.shape[1] / 10))  # figsize=(5,10)
        plt.imshow(mprimg, cmap=plt.cm.gray)
        # plt.colorbar()
        lwidth = 2
        plt.plot(bd1[:, 0], bd1[:, 1], '--r', lw=lwidth)
        plt.plot(snake1[:, 0], snake1[:, 1], '-b', lw=lwidth)
        plt.plot(bd2[:, 0], bd2[:, 1], '--r', lw=lwidth)
        plt.plot(snake2[:, 0], snake2[:, 1], '-b', lw=lwidth)
        plt.show()

    f1 = interp1d(snake1[:, 1], snake1[:, 0])
    f2 = interp1d(snake2[:, 1], snake2[:, 0])

    fy = []
    # segvw = np.zeros((pradout.shape[0],pradout.shape[1]))
    for ptidi in range(snake.NP):
        #snake length is changed, upper limit is original length
        fy.append(min(snake1[-1][1], snake.getAccLen(ptidi)))

    intsnake1 = f1(fy)
    intsnake2 = f2(fy)

    for ptidi in range(snake.NP):
        # print(ptidi,icafem.snakelist[snakeid][ptidi].rad,(intsnake1[ptidi]-intsnake2[ptidi])/2)
        snake[ptidi].rad = (intsnake1[ptidi] - intsnake2[ptidi]) / 2
