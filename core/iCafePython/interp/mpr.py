import numpy as np
import math
import matplotlib.pyplot as plt
from ..point3d import Point3D
from .interp_utils import getCSPos
# from .cs import get_rot_matrix
from rich import print

from .cs import vecs_basis_xsec, sample_array


def grids_polar_from_bases(uvs, n_angles: int = 120, n_points_dia: int = 100):
    dias = np.arange(n_points_dia) # [D]
    dias = dias - (n_points_dia - 1) / 2
    thetas = np.linspace(0, 2 * np.pi, n_angles) # [A]
    thetas, dias = np.meshgrid(thetas, dias) # [D, A] * 2
    # do not need to flip dias along axis 0, because xs should increase
    xs, ys = dias * np.cos(thetas), dias * np.sin(thetas) # [D, A] * 2
    xys = np.stack([xs, ys], axis=-1) # [D, A, 2]
    xys = xys[..., None] # [D, A, 2, 1]
    uvs = uvs[:, None, None] # [P, 1, 1, 2, 3]
    grids = np.sum(xys * uvs, axis=-2) #[P, D, A, 3]
    return grids


def mprStack(self, snake, vrange=50, src='o'):
    curve = np.empty((snake.NP, 3)) # [P, 3]
    for i in range(snake.NP):
        pos = snake[i].pos
        curve[i] = [pos.x, pos.y, pos.z]
    uvs = vecs_basis_xsec(curve) # [P, 2, 3]
    grids = grids_polar_from_bases(uvs, n_angles=120, n_points_dia=2 * vrange)
    grids = grids + curve[:, None, None]
    if src in self.posRTMat:
        rtm = np.asarray(self.posRTMat[src])
        assert rtm.shape == (4, 3)
        rtm = np.concatenate([rtm[:3], rtm[3].reshape(3, 1)], axis=-1) # [3, 4]
        assert rtm.shape == (3, 4)
        grids = np.concatenate([grids, np.ones((*grids.shape[:-1], 1))], axis=-1) # [P, D, A, 4]
        grids = rtm[None, ...] @ grids[..., None] # [P, D, A, 3, 1]
        grids = grids[..., 0] # [P, D, A, 3]
    if self.rzratio != 1:
        grids[..., -1] = grids[..., -1] / self.rzratio
    assert len(self.I[src].shape) == 3

    cs_stack = sample_array(self.I[src], grids) # [P, D, A]
    return cs_stack


def mpr(self, snake, mode='s', src='o', vrange=50, rot=0, exportcpos=0):
    if type(snake) == int and snake >= 0 and snake < len(self.snakelist):
        snake = self.snakelist[snake]
    if mode == 's':
        # straightened reformation
        return self._generateMPRSnake(snake, vrange, rot, src)
    elif mode == 'c':
        # curve reformation
        return self._generateCPRSnake(snake, vrange, rot, src)
    elif mode == 'p':
        # projection curve reformation
        return self._generateCPPRSnake(snake, vrange, rot, src, exportcpos)
    else:
        raise TypeError('Unknown mode')


def _generateMPRSnake(self, snake, vrange=50, rot=0, src='o'):
    #generate straightened cpr using given centerline
    #vrange: profile range in cross sectional plane
    #rot: rotated angle in degree
    #src: which image src to use for mpr
    mprslices = []
    resampled_snake = snake.resampleSnake(1)
    resampled_snake = resampled_snake.movingAvgSnake()

    for ptid in range(len(resampled_snake)):
        cnorm = resampled_snake.getNorm(ptid)
        cpos = resampled_snake[ptid].pos
        cmpr_slice = np.zeros((2 * vrange))
        for rho in range(-vrange, vrange):
            u = rho * math.cos(rot / 180 * np.pi)
            v = rho * math.sin(rot / 180 * np.pi)
            cmpr_slice[vrange + rho] = self.getInt(getCSPos(cnorm, cpos, u, v), src=src)
        mprslices.append(cmpr_slice)
    return np.array(mprslices)


# def _generateMPRSnake(self, snake, vrange=50, rot=0, src='o'):
#     #generate straightened cpr using given centerline
#     #vrange: profile range in cross sectional plane
#     #rot: rotated angle in degree
#     #src: which image src to use for mpr
#     mprslices = []
#     resampled_snake = snake.resampleSnake(1)
#     resampled_snake = resampled_snake.movingAvgSnake()
#
#     norm_uv_pre = None
#     for ptid in range(len(resampled_snake)):
#         norm = resampled_snake.getNorm(ptid)
#         pos = resampled_snake[ptid].pos
#         norm = np.asarray([norm.x, norm.y, norm.z])
#         norm = norm / np.linalg.norm(norm)
#         if norm_uv_pre is None:
#             # [3, 3], norm vector + two unit basis vectors in cross-sectional plane
#             norm_uv_pre = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         rot_matrix = get_rot_matrix(norm_uv_pre[0], norm) # [3, 3]
#         norm_uv = rot_matrix @ np.expand_dims(norm_uv_pre, axis=-1) # [3, 3, 1]
#         norm_uv = norm_uv[..., 0] # [3, 3]
#         norm_uv = norm_uv / np.linalg.norm(norm_uv, axis=-1, keepdims=True)
#         norm_uv_pre = norm_uv
#         # assert np.allclose(norm_uv[0], norm), [norm_uv[0], norm]
#
#         rho = np.arange(-vrange, vrange) # [B]
#         x, y = rho * math.cos(rot / 180 * np.pi), rho * math.sin(rot / 180 * np.pi) # [B] x 2
#         x, y = np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1) # [B, 1] x 2
#         x_rotated, y_rotated = x * norm_uv[1], y * norm_uv[2] # [B, 3] x 2
#         pos_rotated = x_rotated + y_rotated # [B, 3]
#         pos_rotated = pos_rotated + np.array([pos.x, pos.y, pos.z])
#
#         cmpr_slice = np.zeros((2 * vrange))
#         for i in range(2 * vrange):
#             cmpr_slice[i] = self.getInt(Point3D(pos_rotated[i]), src)
#         # for rho in range(-vrange, vrange):
#         #     u = rho * math.cos(rot / 180 * np.pi)
#         #     v = rho * math.sin(rot / 180 * np.pi)
#         #     cmpr_slice[vrange + rho] = self.getInt(getCSPos(norm, pos, u, v), src=src)
#         mprslices.append(cmpr_slice)
#     return np.array(mprslices)


def _generateCPRSnake(self, snake, vrange=50, rot=0, src='o'):
    #stretch curves in centerline, y axis of cpr is the length of centerline
    #vrange: padding to cmpr
    #rot: rotation angle
    #which image src to use for mpr
    ptnum = snake.NP
    rangepos = [i.pos.x * math.cos(rot / 180 * np.pi) + i.pos.y * math.sin(rot / 180 * np.pi) for i in snake]
    xmin = int(np.floor(np.min(rangepos) - vrange))
    xmax = int(np.ceil(np.max(rangepos) + vrange))
    cprslices = []
    straightpos = []
    acclen = 0
    for ptid in range(ptnum - 1):
        norm = snake.getNorm(ptid)
        if ptid == ptnum - 2:
            nextnorm = norm
        else:
            nextnorm = snake.getNorm(ptid + 1)
        lpos = snake[ptid].pos
        npos = snake[ptid + 1].pos
        clen = npos.dist(lpos)
        for ipti in range(int(np.ceil(clen) + 1)):
            if np.ceil(acclen) + ipti >= acclen + clen and ptid != ptnum - 2 or np.ceil(
                    acclen) + ipti > acclen + clen and ptid == ptnum - 2:
                acclen += clen
                #print('acclen',acclen,clen)
                break
            normforward = np.ceil(acclen) + ipti - acclen
            cpos = lpos + norm * (normforward / clen)
            #use weighted norm to avoid sudden change in norm direction
            cnorm = Point3D(0, 0, 1)
            #print(len(cprslices),ipti,clen,normforward,cpos,cnorm)
            straightpos.append(cpos)
            cprslice = np.zeros((xmax - xmin))
            for posx in range(xmin, xmax):
                rho = cpos.x * math.cos(rot / 180 * np.pi) + cpos.y * math.sin(rot / 180 * np.pi) - posx
                u = rho * math.cos(rot / 180 * np.pi)
                v = rho * math.sin(rot / 180 * np.pi)
                #pos,self.getcspos(norm,pos,u,v),
                cprslice[posx - xmin] = self.getInt(getCSPos(cnorm, cpos, u, v), src=src)
            cprslices.append(cprslice)
    return np.array(cprslices)


def _generateCPPRSnake(self, snake, vrange=50, rot=0, src='o', exportcpos=0):
    #projection curve planar reformation
    #vrange: padding to cmpr
    #rot: rotation angle
    #which image src to use for mpr
    ptnum = snake.NP
    rangepos = [i.pos.x * math.cos(rot / 180 * np.pi) + i.pos.y * math.sin(rot / 180 * np.pi) for i in snake]
    zrangepos = [i.pos.z for i in snake]
    xmin = int(np.floor(np.min(rangepos) - vrange))
    xmax = int(np.ceil(np.max(rangepos) + vrange))
    zmin = np.min(zrangepos)
    zrange = int(np.ceil(np.max(zrangepos) - np.min(zrangepos)))
    #print('xrange',xmin,xmax,zmin)
    cprslices = np.zeros((zrange, xmax - xmin))
    straightpos = []
    acclen = 0
    for ptid in range(ptnum - 1):
        norm = snake.getNorm(ptid)
        if ptid == ptnum - 2:
            nextnorm = norm
        else:
            nextnorm = snake.getNorm(ptid + 1)
        lpos = snake[ptid].pos
        npos = snake[ptid + 1].pos
        clen = npos.dist(lpos)
        zstart = min(lpos.z, npos.z)
        zstartint = int(np.ceil(zstart - zmin))
        zend = max(lpos.z, npos.z)
        zendint = int(np.ceil(zend - zmin))
        if zstart == lpos.z:
            startpos = lpos
            endpos = npos
        else:
            startpos = npos
            endpos = lpos
        for intptzi in range(zstartint, zendint):
            if np.sum(cprslices[intptzi]) != 0:
                continue
            zforward = intptzi - (zstart - zmin)
            cpos = startpos + (endpos - startpos) * zforward / (zend - zstart)

            #print(intptzi,'in range',zstartint,zendint,'cposz',cpos.z,'zforward',zforward)
            #use weighted norm to avoid sudden change in norm direction
            cnorm = Point3D(0, 0, 1)
            #print(len(cprslices),ipti,clen,normforward,cpos,cnorm)
            straightpos.append(cpos)
            for posx in range(xmin, xmax):
                rho = cpos.x * math.cos(rot / 180 * np.pi) + cpos.y * math.sin(rot / 180 * np.pi) - posx
                u = rho * math.cos(rot / 180 * np.pi)
                v = rho * math.sin(rot / 180 * np.pi)
                #pos,self.getcspos(norm,pos,u,v),
                cprslices[intptzi, posx - xmin] = self.getInt(getCSPos(cnorm, cpos, u, v), src=src)
    if exportcpos:
        return [np.array(cprslices), straightpos]
    else:
        return np.array(cprslices)


# def mprStack_ori(self, snake, vrange=50, src='o'):
#     if type(snake) == int:
#         snake = self.snakelist[snake]
#     mpr_stack = np.zeros((snake.resampleSnake(1).NP, 2 * vrange, 120))
#     for roti in range(120):
#         #if roti%10==0:
#         #print('\rGenerating MPR',roti*3,'/',360,end='')
#         mpr_stack[:, :, roti] = self.mpr(snake, mode='s', src=src, vrange=vrange, rot=roti * 3)
#     return mpr_stack


def showMPRSnake(self, tempsnake, rot=0, src='o'):
    mprimg = self._generateMPRSnake(tempsnake, rot=0 + rot, src=src)
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for ptidi in range(len(tempsnake)):
        if ptidi % 1 == 0 or ptidi == len(tempsnake) - 1:
            x1.append(mprimg.shape[1] // 2 + tempsnake[ptidi].rad)
            y1.append(tempsnake.getAccLen(ptidi))
            x2.append(mprimg.shape[1] // 2 - tempsnake[ptidi].rad)
            y2.append(tempsnake.getAccLen(ptidi))
    bd1 = np.array([x1, y1]).T
    bd2 = np.array([x2, y2]).T

    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.title('MPR rotate 0')
    plt.imshow(mprimg, cmap='gray')
    lwidth = 2
    plt.plot(bd1[:, 0], bd1[:, 1], '--r', lw=lwidth)
    #plt.plot(snake1[:, 0], snake1[:, 1], '-b', lw=lwidth)
    plt.plot(bd2[:, 0], bd2[:, 1], '--r', lw=lwidth)
    #plt.plot(snake2[:, 0], snake2[:, 1], '-b', lw=lwidth)

    mprimg = self._generateMPRSnake(tempsnake, rot=90 + rot, src=src)
    plt.subplot(1, 2, 2)
    plt.title('MPR rotate 90')
    plt.imshow(mprimg, cmap='gray')
    plt.plot(bd1[:, 0], bd1[:, 1], '--r', lw=lwidth)
    #plt.plot(snake1[:, 0], snake1[:, 1], '-b', lw=lwidth)
    plt.plot(bd2[:, 0], bd2[:, 1], '--r', lw=lwidth)
    #plt.plot(snake2[:, 0], snake2[:, 1], '-b', lw=lwidth)
    #plt.colorbar()
    plt.show(block=False)


'''
import numpy as np
import math
import matplotlib.pyplot as plt
from ..point3d import Point3D
from .interp_utils import getCSPos

def mpr(self, snake, mode='s', src='o', vrange=50, rot=0, exportcpos=0):
	if type(snake) == int and snake >= 0 and snake < len(self.snakelist):
		snake = self.snakelist[snake]
	if mode == 's':
		# straightened reformation
		return self._generateMPRSnake(snake, vrange, rot, src)
	elif mode == 'c':
		# curve reformation
		return self._generateCPRSnake(snake, vrange, rot, src)
	elif mode == 'p':
		# projection curve reformation
		return self._generateCPPRSnake(snake, vrange, rot, src, exportcpos)
	else:
		raise TypeError('Unknown mode')


def _generateMPRSnake(self, snake, vrange=50, rot=0, src='o'):
	#generate straightened cpr using given centerline
	#vrange: profile range in cross sectional plane
	#rot: rotated angle in degree
	#src: which image src to use for mpr
	mprslices = []
	resampled_snake = snake.resampleSnake(1)
	resampled_snake = resampled_snake.movingAvgSnake()

	for ptid in range(len(resampled_snake)):
		cnorm = resampled_snake.getNorm(ptid)
		cpos = resampled_snake[ptid].pos
		cmpr_slice = np.zeros((2*vrange))
		for rho in range(-vrange,vrange):
			u = rho * math.cos(rot/180*np.pi)
			v = rho * math.sin(rot/180*np.pi)
			cmpr_slice[vrange+rho] = self.getInt(getCSPos(cnorm,cpos,u,v),src=src)
		mprslices.append(cmpr_slice)
	return np.array(mprslices)


def _generateCPRSnake(self, snake, vrange=50, rot=0, src='o'):
	#stretch curves in centerline, y axis of cpr is the length of centerline
	#vrange: padding to cmpr
	#rot: rotation angle
	#which image src to use for mpr
	ptnum = snake.NP
	rangepos = [i.pos.x*math.cos(rot/180*np.pi)+i.pos.y*math.sin(rot/180*np.pi) for i in snake] 
	xmin = int(np.floor(np.min(rangepos)-vrange))
	xmax = int(np.ceil(np.max(rangepos)+vrange))
	cprslices = []
	straightpos = []
	acclen = 0
	for ptid in range(ptnum-1):
		norm = snake.getNorm(ptid)
		if ptid == ptnum-2:
			nextnorm = norm
		else:
			nextnorm = snake.getNorm(ptid + 1)
		lpos = snake[ptid].pos
		npos = snake[ptid+1].pos
		clen = npos.dist(lpos)
		for ipti in range(int(np.ceil(clen)+1)):
			if np.ceil(acclen)+ipti>=acclen+clen and ptid!=ptnum-2 or np.ceil(acclen)+ipti>acclen+clen and ptid==ptnum-2:
				acclen += clen
				#print('acclen',acclen,clen)
				break
			normforward = np.ceil(acclen)+ipti-acclen
			cpos = lpos + norm*(normforward/clen)
			#use weighted norm to avoid sudden change in norm direction 
			cnorm = Point3D(0,0,1)
			#print(len(cprslices),ipti,clen,normforward,cpos,cnorm)
			straightpos.append(cpos)
			cprslice = np.zeros((xmax-xmin))
			for posx in range(xmin,xmax):
				rho = cpos.x*math.cos(rot/180*np.pi)+cpos.y*math.sin(rot/180*np.pi)-posx
				u = rho * math.cos(rot/180*np.pi)
				v = rho * math.sin(rot/180*np.pi)
				#pos,self.getcspos(norm,pos,u,v),
				cprslice[posx-xmin] = self.getInt(getCSPos(cnorm,cpos,u,v),src=src)
			cprslices.append(cprslice)
	return np.array(cprslices)


def _generateCPPRSnake(self, snake, vrange=50, rot=0, src='o', exportcpos=0):
	#projection curve planar reformation
	#vrange: padding to cmpr
	#rot: rotation angle
	#which image src to use for mpr
	ptnum = snake.NP
	rangepos = [i.pos.x*math.cos(rot/180*np.pi)+i.pos.y*math.sin(rot/180*np.pi) for i in snake] 
	zrangepos = [i.pos.z for i in snake] 
	xmin = int(np.floor(np.min(rangepos)-vrange))
	xmax = int(np.ceil(np.max(rangepos)+vrange))
	zmin = np.min(zrangepos)
	zrange = int(np.ceil(np.max(zrangepos)-np.min(zrangepos)))
	#print('xrange',xmin,xmax,zmin)
	cprslices = np.zeros((zrange,xmax-xmin))
	straightpos = []
	acclen = 0
	for ptid in range(ptnum-1):
		norm = snake.getNorm(ptid)
		if ptid == ptnum-2:
			nextnorm = norm
		else:
			nextnorm = snake.getNorm(ptid + 1)
		lpos = snake[ptid].pos
		npos = snake[ptid+1].pos
		clen = npos.dist(lpos)
		zstart = min(lpos.z,npos.z)
		zstartint = int(np.ceil(zstart-zmin))
		zend = max(lpos.z,npos.z)
		zendint = int(np.ceil(zend-zmin))
		if zstart == lpos.z:
			startpos = lpos
			endpos = npos
		else:
			startpos = npos
			endpos = lpos
		for intptzi in range(zstartint,zendint):
			if np.sum(cprslices[intptzi])!=0:
				continue
			zforward = intptzi - (zstart-zmin)
			cpos = startpos + (endpos-startpos)*zforward/(zend-zstart)

			#print(intptzi,'in range',zstartint,zendint,'cposz',cpos.z,'zforward',zforward)
			#use weighted norm to avoid sudden change in norm direction 
			cnorm = Point3D(0,0,1)
			#print(len(cprslices),ipti,clen,normforward,cpos,cnorm)
			straightpos.append(cpos)
			for posx in range(xmin,xmax):
				rho = cpos.x*math.cos(rot/180*np.pi)+cpos.y*math.sin(rot/180*np.pi)-posx
				u = rho * math.cos(rot/180*np.pi)
				v = rho * math.sin(rot/180*np.pi)
				#pos,self.getcspos(norm,pos,u,v),
				cprslices[intptzi,posx-xmin] = self.getInt(getCSPos(cnorm,cpos,u,v),src=src)
	if exportcpos:
		return [np.array(cprslices),straightpos]
	else:
		return np.array(cprslices)


def mprStack(self,snake,vrange=50, src='o'):
	if type(snake)==int:
		snake = self.snakelist[snake]
	mpr_stack = np.zeros((snake.resampleSnake(1).NP,2*vrange,120))
	for roti in range(120):
		#if roti%10==0:
			#print('\rGenerating MPR',roti*3,'/',360,end='')
		mpr_stack[:,:,roti] = self.mpr(snake,mode='s',src=src,vrange=vrange,rot=roti*3)
	return mpr_stack


def showMPRSnake(self,tempsnake,rot=0,src='o'):
	mprimg = self._generateMPRSnake(tempsnake, rot=0 + rot, src=src)
	x1 = []
	y1 = []
	x2 = []
	y2 = []

	for ptidi in range(len(tempsnake)):
		if ptidi%1==0 or ptidi == len(tempsnake)-1:
			x1.append(mprimg.shape[1]//2+tempsnake[ptidi].rad)
			y1.append(tempsnake.getAccLen(ptidi))
			x2.append(mprimg.shape[1]//2-tempsnake[ptidi].rad)
			y2.append(tempsnake.getAccLen(ptidi))
	bd1 = np.array([x1,y1]).T
	bd2 = np.array([x2,y2]).T

	plt.figure(figsize=(10,20))
	plt.subplot(1,2,1)
	plt.title('MPR rotate 0')
	plt.imshow(mprimg,cmap='gray')
	lwidth = 2
	plt.plot(bd1[:, 0], bd1[:, 1], '--r', lw=lwidth)
	#plt.plot(snake1[:, 0], snake1[:, 1], '-b', lw=lwidth)
	plt.plot(bd2[:, 0], bd2[:, 1], '--r', lw=lwidth)
	#plt.plot(snake2[:, 0], snake2[:, 1], '-b', lw=lwidth)
	
	mprimg = self._generateMPRSnake(tempsnake, rot=90 + rot, src=src)
	plt.subplot(1,2,2)
	plt.title('MPR rotate 90')
	plt.imshow(mprimg,cmap='gray')
	plt.plot(bd1[:, 0], bd1[:, 1], '--r', lw=lwidth)
	#plt.plot(snake1[:, 0], snake1[:, 1], '-b', lw=lwidth)
	plt.plot(bd2[:, 0], bd2[:, 1], '--r', lw=lwidth)
	#plt.plot(snake2[:, 0], snake2[:, 1], '-b', lw=lwidth)
	#plt.colorbar()
	plt.show()
'''
