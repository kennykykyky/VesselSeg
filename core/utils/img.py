import numpy as np
import cv2
from rich import print
import copy

def croppatch(cartimgori, cty=-1, ctx=-1, sheight=40, swidth=40, include_center=False):
    if len(cartimgori.shape) == 2:
        pxtype = type(cartimgori[0, 0])
    elif len(cartimgori.shape) == 3:
        pxtype = type(cartimgori[0, 0, 0])
    elif len(cartimgori.shape) == 4:
        pxtype = type(cartimgori[0, 0, 0, 0])
    else:
        raise TypeError('dimension is not 2-4')
    cartimgori = copy.copy(cartimgori)

    def croppatch3(cartimgori, cty=-1, ctx=-1, sheight=40, swidth=40, include_center=False):
        #input height, width, (channel) large image, and a patch center position (cty,ctx)
        #output sheight, swidth, (channel) patch with padding zeros
        sheight = int(round(sheight))
        swidth = int(round(swidth))
        patchheight = sheight * 2
        patchwidth = swidth * 2
        if include_center:
            include_center = 1
        else:
            include_center = 0

        patchheight += include_center
        patchwidth += include_center
        if len(cartimgori.shape) < 2:
            print('Not enough dim')
            return
        elif len(cartimgori.shape) == 2:
            cartimg = cartimgori[:, :, None]
        elif len(cartimgori.shape) == 3:
            cartimg = cartimgori
        elif len(cartimgori.shape) > 3:
            print('Too many dim')
            return

        patchchannel = cartimg.shape[2]

        inputheight = cartimg.shape[0]
        inputwidth = cartimg.shape[1]
        #if no center point defined, use mid of cartimg
        if cty == -1:
            cty = inputheight // 2
        if ctx == -1:
            ctx = inputwidth // 2
        ctx = int(round(ctx))
        cty = int(round(cty))
        if ctx - swidth > cartimgori.shape[1] or cty - sheight > cartimgori.shape[0]:
            print('center outside patch', ctx - swidth, cartimgori.shape[1], cty - sheight, cartimgori.shape[0])
            cartimgcrop = np.zeros((patchheight, patchwidth, patchchannel), dtype=pxtype)
            return cartimgcrop
        #crop start end position
        if ctx - swidth < 0:
            p1 = 0
            r1 = -(ctx - swidth)
        else:
            p1 = ctx - swidth
            r1 = 0
        if ctx + swidth + include_center > inputwidth:
            p2 = inputwidth
            r2 = (ctx + swidth + include_center) - inputwidth
        else:
            p2 = ctx + swidth + include_center
            r2 = 0
        if cty - sheight < 0:
            p3 = 0
            r3 = -(cty - sheight)
        else:
            p3 = cty - sheight
            r3 = 0
        if cty + sheight + include_center > inputheight:
            p4 = inputheight
            r4 = (cty + sheight + include_center) - inputheight
        else:
            p4 = cty + sheight + include_center
            r4 = 0
        cartimgcrop = cartimg[p3:p4, p1:p2]
        #if not enough to extract, pad zeros at end
        if cartimgcrop.shape != (patchheight, patchwidth, patchchannel):
            #print('Label Extract region out of border',p1,p2,p3,p4,r1,r2,r3,r4)
            cartimgcropc = cartimgcrop.copy()
            cartimgcrop = np.zeros((patchheight, patchwidth, patchchannel), dtype=pxtype)
            cartimgcrop[r3:patchheight - r4, r1:patchwidth - r2] = cartimgcropc

        if len(cartimgori.shape) == 2:
            return cartimgcrop[:, :, 0]
        else:
            return cartimgcrop

        return cartimgcrop

    if len(cartimgori.shape) < 4:
        return croppatch3(cartimgori, cty, ctx, sheight, swidth, include_center)
    elif len(cartimgori.shape) > 4:
        print('Too many dim')
        return
    else:
        inputdepth = cartimgori.shape[2]
        cartimgcrop = np.zeros((sheight * 2, swidth * 2, inputdepth, cartimgori.shape[3]), dtype=pxtype)
        for dpi in range(inputdepth):
            cartimgcrop[:, :, dpi, :] = croppatch3(cartimgori[:, :, dpi, :], cty, ctx, sheight, swidth, include_center)
        return cartimgcrop

def croppatch3d(cartimgori, cty=-1, ctx=-1, ctz=-1, sheight=8, swidth=8, sdepth=8):
    if ctz == -1:
        ctz = cartimgori.shape[2] // 2
    assert ctx < cartimgori.shape[1]
    assert cty < cartimgori.shape[0]
    assert ctz < cartimgori.shape[2]
    ctz = int(round(ctz))

    cartpatch = croppatch(cartimgori, cty=cty, ctx=ctx, sheight=sheight, swidth=swidth)
    if len(cartimgori.shape) == 3:
        pxtype = type(cartimgori[0, 0, 0])
    elif len(cartimgori.shape) == 4:
        pxtype = type(cartimgori[0, 0, 0, 0])
    else:
        raise TypeError('dimension not 3/4')
    if ctz < sdepth:
        if len(cartimgori.shape) == 3:
            padcartpatch = np.zeros((sheight * 2, swidth * 2, sdepth * 2), dtype=pxtype)
        elif len(cartimgori.shape) == 4:
            padcartpatch = np.zeros((sheight * 2, swidth * 2, sdepth * 2, cartimgori.shape[3]), dtype=pxtype)
        padcartpatch[:, :, sdepth - ctz:] = cartpatch[:, :, :ctz + sdepth]
        return padcartpatch
    elif ctz > cartimgori.shape[2] - sdepth:
        if len(cartimgori.shape) == 3:
            padcartpatch = np.zeros((sheight * 2, swidth * 2, sdepth * 2), dtype=pxtype)
        elif len(cartimgori.shape) == 4:
            padcartpatch = np.zeros((sheight * 2, swidth * 2, sdepth * 2, cartimgori.shape[3]), dtype=pxtype)
        padcartpatch[:, :, :sdepth + cartimgori.shape[2] - ctz] = cartpatch[:, :, ctz - sdepth:]
        return padcartpatch
    else:
        return cartpatch[:, :, ctz - sdepth:ctz + sdepth]