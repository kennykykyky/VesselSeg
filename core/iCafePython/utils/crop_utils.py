import numpy as np
import cv2

def fillpatch(srcimg,patchimg,cty=-1,ctx=-1):
    sheight = patchimg.shape[0]//2
    swidth = patchimg.shape[1]//2
    sheightRem = patchimg.shape[0] - patchimg.shape[0]//2
    swidthRem = patchimg.shape[1] - patchimg.shape[1]//2

    fillimg = srcimg.copy()
    inputheight = srcimg.shape[0]
    inputwidth = srcimg.shape[1]
    patchheight = patchimg.shape[0]
    patchwidth = patchimg.shape[1]

    if cty==-1:
        cty = inputheight//2
    if ctx==-1:
        ctx = inputwidth//2
    ctx = int(round(ctx))
    cty = int(round(cty))

    if ctx-swidth<0:
        p1 = 0
        r1 = -(ctx-swidth)
    else:
        p1 = ctx-swidth
        r1 = 0
    if ctx+swidthRem>inputwidth:
        p2 = inputwidth
        r2 = (ctx+swidthRem)-inputwidth
    else:
        p2 = ctx+swidthRem
        r2 = 0
    if cty-sheight<0:
        p3 = 0
        r3 = -(cty-sheight)
    else:
        p3 = cty-sheight
        r3 = 0
    if cty+sheightRem>inputheight:
        p4 = inputheight
        r4 = (cty+sheightRem)-inputheight
    else:
        p4 = cty+sheightRem
        r4 = 0
    #print(p1,p2,p3,p4,r1,r2,r3,r4)
    fillimg[p3:p4,p1:p2] = patchimg[r3:patchheight-r4,r1:patchwidth-r2]

    return fillimg

def croppatch3d(cartimgori,cty=-1,ctx=-1,ctz=-1,sheight=8,swidth=8,sdepth=8):
    if ctz==-1:
        ctz = cartimgori.shape[2]//2
    assert ctx<cartimgori.shape[0]
    assert cty<cartimgori.shape[1]
    assert ctz<cartimgori.shape[2]
    ctz = int(round(ctz))

    cartpatch = croppatch(cartimgori,cty=cty,ctx=ctx,sheight=sheight,swidth=swidth)
    if len(cartimgori.shape)==3:
        pxtype = type(cartimgori[0, 0, 0])
    elif len(cartimgori.shape)==4:
        pxtype = type(cartimgori[0, 0, 0, 0])
    else:
        raise TypeError('dimension not 3/4')
    if ctz<sdepth:
        if len(cartimgori.shape) == 3:
            padcartpatch = np.zeros((sheight*2,swidth*2,sdepth*2),dtype=pxtype)
        elif len(cartimgori.shape) == 4:
            padcartpatch = np.zeros((sheight*2,swidth*2,sdepth*2,cartimgori.shape[3]),dtype=pxtype)
        padcartpatch[:,:,sdepth-ctz:] = cartpatch[:,:,:ctz+sdepth]
        return padcartpatch
    elif ctz>cartimgori.shape[2]-sdepth:
        if len(cartimgori.shape) == 3:
            padcartpatch = np.zeros((sheight * 2, swidth * 2, sdepth * 2), dtype=pxtype)
        elif len(cartimgori.shape) == 4:
            padcartpatch = np.zeros((sheight * 2, swidth * 2, sdepth * 2, cartimgori.shape[3]), dtype=pxtype)
        padcartpatch[:,:,:sdepth+cartimgori.shape[2]-ctz] = cartpatch[:,:,ctz-sdepth:]
        return padcartpatch
    else:
        return cartpatch[:,:,ctz-sdepth:ctz+sdepth]


import copy
def croppatch(cartimgori,cty=-1,ctx=-1,sheight=40,swidth=40,include_center=False):
    if len(cartimgori.shape) == 2:
        pxtype = type(cartimgori[0,0])
    elif len(cartimgori.shape) == 3:
        pxtype = type(cartimgori[0, 0, 0])
    elif len(cartimgori.shape) == 4:
        pxtype = type(cartimgori[0, 0, 0 ,0])
    else:
        raise TypeError('dimension is not 2-4')
    cartimgori = copy.copy(cartimgori)
    def croppatch3(cartimgori,cty=-1,ctx=-1,sheight=40,swidth=40,include_center=False):
        #input height, width, (channel) large image, and a patch center position (cty,ctx)
        #output sheight, swidth, (channel) patch with padding zeros
        sheight = int(round(sheight))
        swidth = int(round(swidth))
        patchheight = sheight*2
        patchwidth = swidth*2
        if include_center:
            include_center = 1
        else:
            include_center = 0

        patchheight += include_center
        patchwidth += include_center
        if len(cartimgori.shape)<2:
            print('Not enough dim')
            return
        elif len(cartimgori.shape)==2:
            cartimg = cartimgori[:,:,None]
        elif len(cartimgori.shape)==3:
            cartimg = cartimgori
        elif len(cartimgori.shape)>3:
            print('Too many dim')
            return

        patchchannel = cartimg.shape[2]

        inputheight = cartimg.shape[0]
        inputwidth = cartimg.shape[1]
        #if no center point defined, use mid of cartimg
        if cty==-1:
            cty = inputheight//2
        if ctx==-1:
            ctx = inputwidth//2
        ctx = int(round(ctx))
        cty = int(round(cty))
        if ctx-swidth>cartimgori.shape[1] or cty-sheight>cartimgori.shape[0]:
            print('center outside patch')
            cartimgcrop = np.zeros((patchheight, patchwidth, patchchannel),dtype=pxtype)
            return cartimgcrop
        #crop start end position
        if ctx-swidth<0:
            p1 = 0
            r1 = -(ctx-swidth)
        else:
            p1 = ctx-swidth
            r1 = 0
        if ctx+swidth+include_center>inputwidth:
            p2 = inputwidth
            r2 = (ctx+swidth+include_center)-inputwidth
        else:
            p2 = ctx+swidth+include_center
            r2 = 0
        if cty-sheight<0:
            p3 = 0
            r3 = -(cty-sheight)
        else:
            p3 = cty-sheight
            r3 = 0
        if cty+sheight+include_center>inputheight:
            p4 = inputheight
            r4 = (cty+sheight+include_center)-inputheight
        else:
            p4 = cty+sheight+include_center
            r4 = 0
        cartimgcrop = cartimg[p3:p4,p1:p2]
        #if not enough to extract, pad zeros at end
        if cartimgcrop.shape!=(patchheight,patchwidth,patchchannel):
            #print('Label Extract region out of border',p1,p2,p3,p4,r1,r2,r3,r4)
            cartimgcropc = cartimgcrop.copy()
            cartimgcrop = np.zeros((patchheight,patchwidth,patchchannel),dtype=pxtype)
            cartimgcrop[r3:patchheight-r4,r1:patchwidth-r2] = cartimgcropc

        if len(cartimgori.shape)==2:
            return cartimgcrop[:,:,0]
        else:
            return cartimgcrop

        return cartimgcrop

    if len(cartimgori.shape)<4:
        return croppatch3(cartimgori,cty,ctx,sheight,swidth,include_center)
    elif len(cartimgori.shape)>4:
        print('Too many dim')
        return
    else:
        inputdepth = cartimgori.shape[2]
        cartimgcrop = np.zeros((sheight*2,swidth*2,inputdepth,cartimgori.shape[3]),dtype=pxtype)
        for dpi in range(inputdepth):
            cartimgcrop[:,:,dpi,:] = croppatch3(cartimgori[:,:,dpi,:],cty,ctx,sheight,swidth,include_center)
        return cartimgcrop


def croppatch_multi_scale(cartimgori,cty=-1,ctx=-1,sheight=40,swidth=40,scales=[2,4],include_center=False):
    img_patch = croppatch(cartimgori,cty,ctx,sheight,swidth,include_center)
    scale_imgs = [img_patch[None]]
    for scale in scales:
        patch_scale = croppatch(cartimgori, cty, ctx, scale * sheight, scale * swidth)
        img_rz = cv2.resize(patch_scale, (0, 0), fx=1 / scale, fy=1 / scale)
        scale_imgs.append(img_rz[None])
    return np.concatenate(scale_imgs, axis=0)
