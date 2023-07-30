from .centerline_net import CenterlineNet
from ..point3d import Point3D
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
from .ocs import initOcsOptions
from .ocsnake import OCSnake

#ocs+cnn tracker tracing for snakelist
def OCSCNNTracing(traced_snakelist, icafem, infer_model, device, cnntracker_prob_thr,
                  options=None, ref_snake_ids=None):
    starttime = datetime.datetime.now()
    options = initOcsOptions(options)
    traced_snakelist.valid_list = [True] * traced_snakelist.NSnakes
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
        if traced_snakelist.valid_list[snakei] == False:
            continue
        print('\n', '@' * 20, 'Refining Tracing', snakei, '/', len(ref_snake_ids), '@' * 20)
        prob_records = [0] * traced_snakelist[snakei].NP
        snake_id = snakei + 1
        temp_snake, valid = OCSCNNSnakeTracing(traced_snakelist[snakei], icafem, traced_snakelist, options,
                                               infer_model,device,cnntracker_prob_thr, snake_id)
        if valid:
            print('\nstretch from', traced_snakelist[snakei].length, 'to', temp_snake.length)
            traced_snakelist[snakei].snake = [swci for swci in temp_snake.snake]

    endtime = datetime.datetime.now()
    reftime = (endtime - starttime).total_seconds()
    print('time for tracing %.1f minutes' % (reftime / 60))
    return traced_snakelist

#cnn trcker model to control head and tail forces
def OCSCNNSnakeTracing(temp_snake,icafem,traced_snakelist,options,infer_model,device,cnntracker_prob_thr,
                       snake_id=None):
    temp_snake = OCSnake(icafem, traced_snakelist, temp_snake)
    #set infer model to OCSnake, otherwise use last direction instead of CNN to predict stretching direction
    temp_snake.setCNNTracker(infer_model,device,cnntracker_prob_thr)
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