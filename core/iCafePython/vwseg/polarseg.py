import numpy as np

def polarVWSegArtery(self, seg_model, snake, src, startid=0, endid=-1, return_cas = False):
    if type(snake)==int:
        snake = self.snakelist[snake]
    contours = []
    confs = []
    if endid == -1:
        endid = snake.NP
    SCALE = 4 * seg_model.cfg['SCALE']
    hps = seg_model.cfg['width'] // SCALE
    cs_stack = self.csStackRange(snake, startid, endid, hheight=hps, hwidth=hps, src=src)
    polar_stack = seg_model.toPolar(cs_stack)
    print('polar conversion end', polar_stack.shape)
    neih = (seg_model.cfg['depth'] - 1) // 2
    for ptid in range(startid, endid):
        print('\rSegmenting', ptid, '/', endid, end='')
        polar_stack_sel = np.zeros((polar_stack.shape[0], polar_stack.shape[1], neih * 2 + 1))
        for offid in range(-neih, neih + 1):
            cptid = ptid + offid
            if cptid < 0:
                cptid = 0
            if cptid >= endid:
                cptid = endid - 1
            polar_stack_sel[:, :, offid + neih] = polar_stack[:, :, cptid-startid]
        polarbd, polarsd = seg_model.predict(polar_stack_sel)
        cont = seg_model.cartbd
        contours.append(cont)
        polarconsistency = 1 - np.mean(polarsd, axis=0)
        #ccstl = polarconsistency[0]
        #ccstw = polarconsistency[1]
        confs.append(polarconsistency)
    if return_cas:
        return contours, confs, cs_stack
    else:
        return contours, confs

def polarVWSegCS(self, seg_model, snake, ptid, src):
    if type(snake)==int:
        snake = self.snakelist[snake]
    SCALE = 4 * seg_model.cfg['SCALE']
    hps = seg_model.cfg['width'] // SCALE
    cs_stack = self.csStackNei(snake, ptid, (seg_model.cfg['depth'] - 1) // 2, hheight=hps, hwidth=hps, src=src)
    polar_stack = seg_model.toPolar(cs_stack)
    seg_model.predict(polar_stack)
    return seg_model.cartbd
