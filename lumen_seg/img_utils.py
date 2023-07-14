import numpy as np
import cv2
from rich import print


def fillpatch(srcimg, patchimg, cty=-1, ctx=-1):
    sheight = patchimg.shape[0] // 2
    swidth = patchimg.shape[1] // 2
    sheightRem = patchimg.shape[0] - patchimg.shape[0] // 2
    swidthRem = patchimg.shape[1] - patchimg.shape[1] // 2

    fillimg = srcimg.copy()
    inputheight = srcimg.shape[0]
    inputwidth = srcimg.shape[1]
    patchheight = patchimg.shape[0]
    patchwidth = patchimg.shape[1]

    if cty == -1:
        cty = inputheight // 2
    if ctx == -1:
        ctx = inputwidth // 2
    ctx = int(round(ctx))
    cty = int(round(cty))

    if ctx - swidth < 0:
        p1 = 0
        r1 = -(ctx - swidth)
    else:
        p1 = ctx - swidth
        r1 = 0
    if ctx + swidthRem > inputwidth:
        p2 = inputwidth
        r2 = (ctx + swidthRem) - inputwidth
    else:
        p2 = ctx + swidthRem
        r2 = 0
    if cty - sheight < 0:
        p3 = 0
        r3 = -(cty - sheight)
    else:
        p3 = cty - sheight
        r3 = 0
    if cty + sheightRem > inputheight:
        p4 = inputheight
        r4 = (cty + sheightRem) - inputheight
    else:
        p4 = cty + sheightRem
        r4 = 0
    #print(p1,p2,p3,p4,r1,r2,r3,r4)
    fillimg[p3:p4, p1:p2] = patchimg[r3:patchheight - r4, r1:patchwidth - r2]

    return fillimg


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


def croppatch_multi_scale(cartimgori, cty=-1, ctx=-1, sheight=40, swidth=40, scales=[2, 4], include_center=False):
    img_patch = croppatch(cartimgori, cty, ctx, sheight, swidth, include_center)
    scale_imgs = [img_patch[None]]
    for scale in scales:
        patch_scale = croppatch(cartimgori, cty, ctx, scale * sheight, scale * swidth)
        img_rz = cv2.resize(patch_scale, (0, 0), fx=1 / scale, fy=1 / scale)
        scale_imgs.append(img_rz[None])
    return np.concatenate(scale_imgs, axis=0)


def enhance_rot_images(rot_imgs, std=3):
    #minus mean, reduce low/high signal by std, then add mean back
    mean_rot_img = np.mean(rot_imgs, axis=0)
    std_rot_img = np.std(rot_imgs, axis=0) * std
    mc = np.mean(mean_rot_img)
    mean_rot_img -= mc
    for i in range(mean_rot_img.shape[0]):
        for j in range(mean_rot_img.shape[1]):
            if mean_rot_img[i, j] < 0:
                mean_rot_img[i, j] = min(mean_rot_img[i, j] + std_rot_img[i, j], 0)
            else:
                mean_rot_img[i, j] = max(0, mean_rot_img[i, j] - std_rot_img[i, j])
    mean_rot_img += mc
    # mean_rot_img[mean_rot_img<0] = 0
    mean_rot_img = mean_rot_img / np.max(mean_rot_img)
    #plt.figure(figsize=(10, 10))
    #plt.imshow(mean_rot_img, cmap='gray')


def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)


def export_norm(img):
    img -= np.min(img)
    img /= np.max(img)
    img *= 255
    return img.astype(np.uint8)


def circle_merge_background(fg, bg, ct, fore_r, radm=2):
    com_stent = copy.copy(bg)
    for r in range(bg.shape[0]):
        for c in range(bg.shape[1]):
            d = np.sqrt(np.sum([pow(([r, c][dim] - ct[dim]), 2) for dim in range(2)]))
            if d > radm * fore_r:
                continue
            elif d < fore_r:
                com_stent[r, c] = fg[r, c]
            else:
                com_stent[r, c] = (radm * fore_r - d) / ((radm - 1) * fore_r) * fg[r, c] + (d - fore_r) / (
                    (radm - 1) * fore_r) * bg[r, c]
    return com_stent


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rot_imgs(imgs, rot):
    imgs_rot = []
    for framei in range(len(imgs)):
        imgs_rot.append(rotate_image(imgs[framei], rot))
    return imgs_rot


#resize to target size for whole stack
def t_resize_stack(img, targetsz):
    resize_img = np.zeros((img.shape[0], targetsz, targetsz, img.shape[3]))
    for framei in range(img.shape[0]):
        ori_img = img[framei, :, :, 0]
        resize_img[framei, :, :, 0] = t_resize(ori_img, targetsz)
    return resize_img


#resize to target size
def t_resize(ori_img, targetsz):
    rz_img = cv2.resize(ori_img, (0, 0), fx=0.5, fy=0.5)
    if rz_img.shape[0] < targetsz:
        resize_img = np.zeros((targetsz, targetsz))
        resize_img[:rz_img.shape[0], :rz_img.shape[1]] = rz_img
    elif rz_img.shape[0] > targetsz:
        resize_img = rz_img[:targetsz, :targetsz]
    else:
        resize_img = rz_img
    return resize_img


#pad images to fit the min_unit
def padImg(img, min_unit):
    padsz1 = int(np.ceil(img.shape[0] / min_unit) * min_unit)
    padsz2 = int(np.ceil(img.shape[1] / min_unit) * min_unit)
    padsz3 = int(np.ceil(img.shape[2] / min_unit) * min_unit)
    imgpad = np.zeros((padsz1, padsz2, padsz3))
    imgpad[:img.shape[0], :img.shape[1], :img.shape[2]] = img
    return imgpad
