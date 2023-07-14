import math
import numpy as np
from ..point3d import Point3D

def getCSDirect(norm, u, v):
    # get 3d direction for target cs plane
    xn = norm.x
    yn = norm.y
    zn = norm.z

    if yn == 0 and zn == 0:
        xu = 0
        yu = 0
        zu = -u
        xv = 0
        yv = v
        zv = 0
    else:
        srt = math.sqrt((yn * yn + zn * zn) ** 2 + xn * xn * yn * yn + xn * xn * zn * zn)
        xu = (yn * yn + zn * zn) * u / srt
        yu = -xn * yn * u / srt
        zu = -zn * xn * u / srt

        xv = 0
        yv = v * zn / math.sqrt(yn * yn + zn * zn)
        zv = -v * yn / math.sqrt(yn * yn + zn * zn)

    return Point3D([xu, yu + yv, zu + zv])
    '''
    sanity check
    xu*xn+yu*yn+zu*zn,\
    xv*xn+yv*yn+zv*zn,\
    np.linalg.norm([xu,yu,zu]),\
    np.linalg.norm([xv,yv,zv]),\
    xu*xv+yu*yv+zu*zv
    '''


def getCSPos(norm, pos, u, v):
    # get position for cross sectional plane
    return pos + getCSDirect(norm, u, v)


def getNormPos(norm, pos, u, v, alignaxis='x'):
    # get intensity for norm plane (can be multiple planes perpendicular to the cross sectional plane, fix one axis x/y/z to become one 2D normal image)
    norm = norm.norm()
    x0 = pos.x
    y0 = pos.y
    z0 = pos.z
    xn = norm.x
    yn = norm.y
    zn = norm.z
    if alignaxis == 'x':
        srt = math.sqrt(yn * yn + zn * zn)
        xu = 0
        yu = u * zn / srt
        zu = -u * yn / srt
    elif alignaxis == 'y':
        srt = math.sqrt(xn * xn + zn * zn)
        xu = u * zn / srt
        yu = 0
        zu = -u * xn / srt
    elif alignaxis == 'z':
        srt = math.sqrt(yn * yn + xn * xn)
        xu = u * yn / srt
        yu = -u * xn / srt
        zu = 0

    xv = xn * v
    yv = yn * v
    zv = zn * v

    return Point3D([x0 + xu + xv, y0 + yu + yv, z0 + zu + zv])


def normCordPos(norm, pos, vect):
    # get vector position in the cordinate system where norm direction is z
    norm = norm.norm()
    x0 = pos.x
    y0 = pos.y
    z0 = pos.z
    xn = norm.x
    yn = norm.y
    zn = norm.z
    x = vect.x
    y = vect.y
    z = vect.z
    if yn == 0 and zn == 0:
        # same cordinate system
        return vect
    else:
        srt = math.sqrt((yn * yn + zn * zn) ** 2 + xn * xn * yn * yn + xn * xn * zn * zn)
        fxu = -(yn * yn + zn * zn) / srt
        fyu = xn * yn / srt
        fzu = zn * xn / srt

        fxv = 0
        fyv = zn / math.sqrt(yn * yn + zn * zn)
        fzv = -yn / math.sqrt(yn * yn + zn * zn)

        fxw = xn
        fyw = yn
        fzw = zn

        fmat = np.array([[fxu, fxv, fxw, x0], [fyu, fyv, fyw, y0], [fzu, fzv, fzw, z0], [0, 0, 0, 1]])
        invfmat = np.linalg.inv(fmat)
        normmat = np.matmul(invfmat, np.transpose([x, y, z, 1]))

        return Point3D(normmat[:3])



def worldcordpos(norm,pos,vect):
	#get world cordinate vector from the cordinate system where norm direction is z
	norm = norm.norm()
	x0 = pos.x
	y0 = pos.y
	z0 = pos.z
	xn = norm.x
	yn = norm.y
	zn = norm.z
	u = vect.x
	v = vect.y
	w = vect.z
	if yn==0 and zn == 0:
		#same cordinate system
		return vect
	else:
		srt = math.sqrt((yn*yn+zn*zn)**2+xn*xn*yn*yn+xn*xn*zn*zn)
		fxu = -(yn*yn+zn*zn)/srt
		fyu = xn*yn/srt
		fzu = zn*xn/srt

		fxv = 0
		fyv = zn/math.sqrt(yn*yn+zn*zn)
		fzv = -yn/math.sqrt(yn*yn+zn*zn)

		fxw = xn
		fyw = yn
		fzw = zn

		posx = fxu*u+fxv*v+fxw*w+x0
		posy = fyu*u+fyv*v+fyw*w+y0
		posz = fzu*u+fzv*v+fzw*w+z0
		return Point3D([posx,posy,posz])
