#open curve snake
from .ocsnake import OCSnake
from ..snake import Snake
from ..snakelist import SnakeList
from ..swcnode import SWCNode
from ..point3d import Point3D
import numpy as np
import datetime
import matplotlib.pyplot as plt

#OCS tracing from seeds
def openCurveSeedsTracing(seeds,icafem,options=None):
    starttime = datetime.datetime.now()
    options = initOcsOptions(options)
    traced_snakelist = SnakeList()
    traced_snakelist.valid_list = []
    vimg = icafem.loadImg('v')
    icafem.I['v'] = icafem.I['v'] / np.max(icafem.I['v']) * 255
    icafem.loadImg('l', np.zeros(icafem.I['o'].shape,dtype=np.int16))
    icafem.imComputeInitForegroundModel()
    icafem.imComputeInitBackgroundModel()
    print('Img foreground:%.1f+-%.1f, background:%.1f+-%.1f'%(icafem.u1, icafem.sigma1, icafem.u2, icafem.sigma2))
    print('ImgV foreground:%.1f+-%.1f, background:%.1f+-%.1f'%(icafem.uv1, icafem.sigmav1, icafem.uv2, icafem.sigmav2))

    seeds_valid = np.ones(len(seeds))
    for seed_pos_i in range(len(seeds)):
        if seeds_valid[seed_pos_i]==0:
            continue
        seed_pos = seeds[seed_pos_i]
        if icafem.getInt(seed_pos, 'l') != 0:
            continue
        print('\n','@'*20,'Tracing from seed',seed_pos,'remaining seeds',np.sum(seeds_valid),'@'*20)
        #from seed to three point snake
        temp_snake = initSnakePoint(seed_pos, icafem, traced_snakelist)
        temp_snake,valid = openCurveSnakeTracing(temp_snake, icafem, traced_snakelist, options, traced_snakelist.NSnakes+1)
        if valid:
            traced_snakelist.addSnake(temp_snake)
            traced_snakelist.valid_list.append(valid)
            print('\ntracing done, length', temp_snake.length, traced_snakelist)

        seeds_valid[seed_pos_i] = 0
        #refresh valid seeds
        for seed_pos_j in range(seed_pos_i,len(seeds)):
            if seeds_valid[seed_pos_j] == 1 and icafem.getInt(seeds[seed_pos_j], 'l') != 0:
                seeds_valid[seed_pos_j] = 0
    endtime = datetime.datetime.now()
    reftime = (endtime - starttime).total_seconds()
    print('time for tracing %.1f minutes'%(reftime/60))
    return traced_snakelist

#refine/extend existing snake using OCS tracing
def openCurveSnakeListRefine(traced_snakelist,icafem,options=None,ref_snake_ids=None):
    starttime = datetime.datetime.now()
    options = initOcsOptions(options)
    traced_snakelist.valid_list = [True]*traced_snakelist.NSnakes
    if 'v' not in icafem.I:
        vimg = icafem.loadImg('v')
    icafem.I['v'] = icafem.I['v'] / np.max(icafem.I['v']) * 255
    icafem.loadImg('l', np.zeros(icafem.I['o'].shape, dtype=np.int16))
    # paint label img
    for snake_i in range(traced_snakelist.NSnakes):
        snake_id = snake_i + 1
        temp_snake = traced_snakelist[snake_i]
        temp_snake.labelImgEncoding(icafem.I['l'], paintVal=snake_id, coding_radius=0.5)
    plt.imshow(np.max(icafem.I['l'], axis=2))
    plt.colorbar()
    plt.show()
    icafem.imComputeInitForegroundModel()
    icafem.imComputeInitBackgroundModel()
    print('Img foreground:%.1f+-%.1f, background:%.1f+-%.1f' % (icafem.u1, icafem.sigma1, icafem.u2, icafem.sigma2))
    print(
        'ImgV foreground:%.1f+-%.1f, background:%.1f+-%.1f' % (icafem.uv1, icafem.sigmav1, icafem.uv2, icafem.sigmav2))

    if ref_snake_ids is None:
        ref_snake_ids = np.arange(traced_snakelist.NSnakes).tolist()
    for snakei in ref_snake_ids:
        if traced_snakelist.valid_list[snakei]==False:
            continue
        print('\n','@' * 20, 'Refining Tracing', snakei, '/', len(ref_snake_ids), '@' * 20)
        snake_id = snakei+1
        temp_snake, valid = openCurveSnakeTracing(traced_snakelist[snakei], icafem, traced_snakelist, options, snake_id)
        if valid:
            print('\nstretch from',traced_snakelist[snakei].length,'to',temp_snake.length)
            traced_snakelist[snakei].snake = [swci for swci in temp_snake.snake]

    endtime = datetime.datetime.now()
    reftime = (endtime - starttime).total_seconds()
    print('time for tracing %.1f minutes' % (reftime / 60))
    return traced_snakelist


def openCurveSnakeTracing(temp_snake,icafem,traced_snakelist,options,snake_id=None):
    #temp_snake is a Snake object

    # for snake_i in range(traced_snakelist.NSnakes):
    #     snake_id = snake_i+1
    #     if traced_snakelist.valid_list[snake_i]==False:
    #         continue
    #     temp_snake = traced_snakelist[snake_i]
    #     temp_snake.labelImgEncoding(icafem.I['l'], paintVal=snake_id, coding_radius=0.5)
    # plt.imshow(np.max(icafem.I['l'], axis=2))
    # plt.colorbar()
    # plt.show()
    temp_snake = OCSnake(icafem, traced_snakelist, temp_snake)
    if snake_id is None:
        snake_id = traced_snakelist.NSnakes+1
    i = 0
    struggle_label = 0
    valid = True
    while i <options['iter_num'] and struggle_label<=options['struggle_th'] and not temp_snake.hit_boundary:
        print('\r','=' * 10, 'iter', i, 'length', temp_snake.length, 'strugle', struggle_label, '=' * 10, end='')
        old_dist = temp_snake.length
        if struggle_label<1:
            vesselnesstracing = options['enable_vesselness_tracing']
        else:
            vesselnesstracing = 0
        temp_snake.openSnakeStretch_4D(options['alpha'], options['stretch_iter'], options['pt_distance'], options['beta'], options['kappa'], options['gamma'],
                                options['stretchingRatio'], options['collision_dist'], options['minimum_length'],
								options['automatic_merging'], options['max_angle'], options['freeze_body'], options['s_force'], snake_id,
								options['tracing_model'], options['coding_method'], options['sigma_ratio'], options['border'], vesselnesstracing)
        #print('temp_snake',temp_snake.snakelist.valid_list)
        if temp_snake.NP < 3:
            valid = False
            break
        new_dist = temp_snake.length
        if new_dist > old_dist * (1 - options['struggle_dist']) and new_dist < old_dist * (1 + options['struggle_dist']):
            struggle_label += 1
        else:
            struggle_label = 0

        #if temp_snake.length > options['minimum_length']:
        #    if temp_snake.NP>5:
        icafem.I['l'][icafem.I['l']==snake_id] = 0
        temp_snake.labelImgEncoding(icafem.I['l'],paintVal=snake_id,coding_radius=0.5)
        # plt.imshow(np.max(icafem.I['l'], axis=2))
        # plt.colorbar()
        # plt.show()
        i += 1

    if temp_snake.length < options['minimum_length']:
        valid = False
        #print(temp_snake.length,'not enough length')

    return temp_snake,valid

def initSnakePoint(pos,icafem,traced_snakelist):
    pos.bound(icafem.SM, icafem.SN, icafem.SZ)
    temp_snake = Snake()
    #default radius 2
    temp_snake.addSWC(pos,2)
    max_intensity = 0
    for i in range(-1,1):
        for j in range(-1, 1):
            for k in range(-1, 1):
                sum_intensity = 0
                temp_p1 = Point3D(pos.x+i,pos.y+j,pos.z+k)
                temp_p2 = Point3D(pos.x-i,pos.y-j,pos.z-k)
                temp_p1.bound(icafem.SM,icafem.SN,icafem.SZ)
                temp_p2.bound(icafem.SM, icafem.SN, icafem.SZ)
                sum_intensity += int(icafem.getInt(temp_p1,'o'))
                sum_intensity += int(icafem.getInt(temp_p2,'o'))
                if sum_intensity > max_intensity and i|j|k != 0:
                    max_intensity = sum_intensity
                    P1 = temp_p1
                    P2 = temp_p2
    if max_intensity == 0:
        print('seed on background')
        P1 = Point3D(pos.x,pos.y,pos.z+1)
        P2 = Point3D(pos.x,pos.y,pos.z-1)
    temp_snake.insert(0,SWCNode(P1,2))
    temp_snake.addSWC(P2,2)
    return temp_snake

def initOcsOptions(options):
    if options is None:
        options = {}
    default_options = {'alpha': 0.05,
     'stretch_iter': 5,
     'pt_distance': 3,
     'beta': 0.05,
     'kappa': 1,
     'gamma': 2.0,
     'stretchingRatio': 3.0,
     'struggle_dist': 0.05,
     'struggle_th': 3,
     'minimum_length': 5.0,
     'iter_num': 100,
     'remove_seed_range': 3,
     'deform_iter': 0,
     'collision_dist': 1,
     'automatic_merging': True,
     'max_angle': 99,
     'freeze_body': False,
     's_force': 1,
     'tracing_model': 1,
     'parallel_tracing': False,
     'coding_method': 1,
     'sigma_ratio': 2.0,
     'border': 0,
     'enable_vesselness_tracing': 1}
    for key in default_options:
        if key not in options:
            options[key] = default_options[key]
    return options