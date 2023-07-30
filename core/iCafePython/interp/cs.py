import numpy as np
import math
from ..point3d import Point3D
from ..swcnode import SWCNode
from ..snake import Snake
from .interp_utils import getCSPos, getNormPos
from rich import print
import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import time


def distance(vecs1: ArrayLike, vecs2: ArrayLike):
    """
    Compute the distances between two group of vectors. Both inputs must be able to be broadcast together.
    Args:
        vecs1: Shape `[..., 3]`, an array of 3D coordinates.
        vecs2: Shape `[..., 3]`, an array of 3D coordinates.
    Returns:
        Array with broadcast shape. E.g., with `vecs1`'s shape `[A, 1, 3]` and `vecs2`'s shape `[1, B, 3]`, 
            the result's shape will be `[A, B]`.
    """
    diff = np.asarray(vecs1) - np.asarray(vecs2)
    return np.asarray(np.sqrt(np.sum(diff**2, axis=-1)))


def distance2d(vecs1: ArrayLike, vecs2: ArrayLike):
    """
    Compute the distances between two group of vectors after projecting them to the x-y plane, i.e., the z coordinate 
    will be regarded as 0. Both inputs must be able to be broadcast together.
    Args:
        vecs1: Shape `[..., 3]`, an array of 3D coordinates.
        vecs2: Shape `[..., 3]`, an array of 3D coordinates.
    Returns:
        Array with broadcast shape. E.g., with `vecs1`'s shape `[A, 1, 3]` and `vecs2`'s shape `[1, B, 3]`, 
            the result's shape will be `[A, B]`.
    """
    diff = np.asarray(vecs1)[..., :2] - np.asarray(vecs2)[..., :2]
    return np.asarray(np.sqrt(np.sum(diff**2, axis=-1)))


def mod(vecs: ArrayLike):
    """
    Compute the moduli of vectors.
    Args:
        vecs: Shape `[..., 3]`, an array of 3D coordinates.
    Returns:
        Shape of `[...]`.
    """
    return distance(vecs, [0, 0, 0])


def normalize(vecs: ArrayLike):
    """
    Normalize vectors to unit length.
    Args:
        vecs: Shape of `[..., 3]`, an array of 3D coordinates.
    Returns:
        Shape of `[..., 3]`, the normalized vectors, i.e., the L2 norms of last dimension are 1. 
            If a vector in `vecs` has length of 0, then the normalized vector will still be a zero vector.
    """
    lengths = mod(vecs)
    return vecs / np.clip(lengths[..., None], a_min=1e-10, a_max=None)


def vecs_tan(curve: ArrayLike, normalized: bool = False):
    r"""
    Get the tangent vectors at points on a curve. Specifically, let $t_i,1\leqslant i\leqslant P$, be the $i$-th tangent
    vector, and $v_i$ be the $i$-th point. Then, $t_1=v_2-v_1,t_P=v_P-v_{P-1}$, and for $2\leqslant i \leqslant P-1$, 
    $t_i=\frac{v_{i+1}-v_i}{\|v_{i+1}-v_i\|}+\frac{v_i-v_{i-1}}{\|v_i-v_{i-1}\|}$. $t_i$'s can be further normalized to 
    unit vectors, if `normalized` is `True`.
    <figure markdown>
    ![girds](../imgs/vecs_tan.png){width="320"}
    </figure>
    Args:
        curve: Shape of `[P, 3]` ($P\geqslant 2$), an array of 3D coordinates representing the points on the curve.
        normalized: If `True`, all tangent vectors are normalized to have unit length.
    Returns:
        vecs: Shape of `[P, 3]`. `vecs[i, 3]` is the 3D coordinates that represent the tangent vector at the i-th point.
    """
    curve = np.asarray(curve)
    diff = curve[1:] - curve[:-1]
    diff = normalize(diff)
    vecs = diff[1:] + diff[:-1]
    vecs = np.concatenate([diff[0].reshape(1, 3), vecs, diff[-1].reshape(1, 3)])
    if normalized:
        vecs = normalize(vecs)
    return vecs


def mats_rot(vecs1: ArrayLike, vecs2: ArrayLike):
    r"""
    Compute the rotation matrixes that rotate vectors in `vecs1` to those in `vecs2`. 
    Specifically, let $x_i$ be the $i$-th vector in `vecs1`, and $y_i$ be the $i$-th vector in `vecs2`, and $R_i$ is 
    the $i$-th rotation matrix in the returned `mats`, then $R_i$ is orthonormal, and 
    $R_i\frac{x_i}{\|x_i\|}=\frac{y_i}{\|y_i\|}$.
    <figure markdown>
    ![girds](../imgs/mats_rot.png){width="250"}
    </figure>
    Args:
        vecs1: Shape `[P, 3]`, an array of 3D coordinates.
        vecs2: Shape `[P, 3]`, an array of 3D coordinates.
    Returns:
        mats: Shape of `[P, 3, 3]`. Array of rotation matrixes, where `mats[i]` is the rotation matrix that rotates 
            `vecs1[i]` to `vecs2[i]`, i.e., `mats` are orthonormal matrixes, and 
            `mats @ normalize(vecs1)[..., None] == normalize(vecs2)[..., None]`.
    """
    # The code is based on Rodrigues' rotation formula described at https://math.stackexchange.com/a/476311
    vecs1, vecs2 = normalize(vecs1), normalize(vecs2)
    p = vecs1.shape[0] # P
    rot = np.cross(vecs1, vecs2) # [P, 3], the rotation axis
    s = mod(rot) # [P]
    c = vecs1[..., None, :] @ vecs2[..., None] # [P, 1, 3] @ [P, 3, 1] == [P, 1, 1]
    zeros = np.zeros([p]) # [P]
    # [P] x 9
    vx = [zeros, -rot[..., 2], rot[..., 1], rot[..., 2], zeros, -rot[..., 0], -rot[..., 1], rot[..., 0], zeros]
    vx = np.stack(vx, axis=-1) # [P, 9]
    vx = vx.reshape(p, 3, 3) # [P, 3, 3]
    # If the two vectors are parallel, s will be 0, thus s need to be cliped.
    mats = np.eye(3) + vx + vx @ vx * (1 - c) / np.clip(s[..., None, None]**2, a_min=1e-10, a_max=None) # [P, 3, 3]
    return mats


def mats_rot_cum(rots: ArrayLike):
    """Compute cumulative rotation matrixs from rotation matrixs.
    Args:
        rots: Shape of `[P, 3, 3]`, a group of $3\times 3$ rotation matrixs.
    Returns:
        Shape of `[P, 3, 3]`, the cumulative rotation matrixs.
    """
    rots_cum = np.empty_like(rots) # [P, 3, 3]
    rots_cum[0] = rots[0]
    for i in range(1, rots.shape[0]):
        rots_cum[i] = rots[i] @ rots_cum[i - 1]
    return rots_cum


def vecs_basis_xsec(curve: ArrayLike = None, tans: ArrayLike = None, rots_cum: ArrayLike = None):
    r"""
    Compute the _orthonormal_ basis (containing two vectors) for the 2D cross-sectional plane at each point on the curve.
    Specifically, let the two basis vectors for the cross-sectional plane of the $i$-th point be $u_i$ and $v_i$, then 
    $\|u_i\|=\|v_i\|=1$, $u_i\bot v_i$, and $u_i, v_i\bot t_i$, where $t_i$ is the ($i$-th) tangent vector of the curve 
    at the $i$-th point. For the definition of the tangent vector, see [aicafe.img.vecs_tan][aicafe.img.vecs_tan].
    Args:
        curve: Shape of `[P, 3]` ($P\geqslant 2$), an array of 3D coordinates representing the points on the curve.
        tans: Shape of `[P, 3]`, an array of 3D coordinates, each of which represents the tangent vector at a point on 
            the curve. If `curve` is not `None`, `tans` will not be used, and tangent vectors will be calculated using
            `curve`.
        rots_cum: Shape of `[P, 3, 3]`, an array of cumulative rotation matrixs. If `None`, it will be calculated from 
            `tans`.
    Returns:
        uvs: Shape of `[P, 2, 3]`. For $1\leqslant i\leqslant P$, `uvs[i-1, 0]` represents $u_i$, and `uvs[i-1, 1]` 
            represents $v_i$.
    Raises:
        ValueError: If `curve` and `tans` are both `None`.
    !!! note "The choice of basis is deterministic."
        There are infinite choices for the basis of a plane, that is, the basis can be rotated along the tangent vector 
        and still be a basis. But in this function the basis of each plane is calculated via a deterministic algorithm:
        1. Initially, we set $t_0=(1,0,0)^T, u_0=(0,1,0)^T,v_0=(0,0,1)^T$.
        2. For the first point on the curve (i.e., $i=1$), we calculate the rotation matrix $R_0$, which rotates $t_0$ 
           to $t_1$. For definition of the rotation matrix, see [aicafe.img.mats_rot][aicafe.img.mats_rot]. Then 
           $u_1=R_0u_0,v_1=R_0v_0$.
        3. For the $i+1$-th ($i\geqslant1$) point, let the rotation matrix that rotates $t_i$ to $t_{i+1}$ be $R_{i}$, 
           then $u_{i+1}=R_iu_i, v_{i+1}=R_iv_i$.
        
        This algorithm minimizes the change in cross-sectional direction at two consecutive points. Thus, if we further
        use the computed basis vectors to generate the cross-sectional images along the curve for a 3D image, the 
        continuity of two consecutive cross-sectional images will be retained to the maximum extent possible, which is 
        essential, for example, for multiplanar reformation (MPR) views in vascular imaging.
    """
    if curve is not None:
        tans = vecs_tan(curve) # [P, 3]
    elif tans is None:
        raise ValueError('one of curve and tans must not be None')
    tans = np.concatenate([[[1, 0, 0]], tans]) # [P+1, 3], add t_0 as the first tangent vector

    if rots_cum is None:
        # [P, 3, 3], compute rotation matrixes between two consecutive tangent vectors
        rots = mats_rot(tans[:-1], tans[1:])
        rots_cum = mats_rot_cum(rots)
    rots_cum = rots_cum[:, None] # [P, 1, 3, 3]

    uv_0 = np.asarray([[0, 1, 0], [0, 0, 1]]) # [2, 3]
    uv_0 = uv_0[None, ..., None] # [1, 2, 3, 1]
    uvs = rots_cum @ uv_0 # [P, 2, 3, 1]
    # [P, 2, 3], although cumulative rotation matrixes are theoretically orthonormal,
    # an additional normalization is still needed to eliminate cumulative error
    uvs = normalize(uvs[..., 0])
    return uvs


def grids_rect_from_bases(uvs: ArrayLike, n_rows: int = 200, n_cols: int = 200, loc_ori: str = 'lower_left'):
    r"""
    Generate rectangular (Cartesian) grids of points from 2D basis vector pairs. 
    Specifically, let $u_k,v_k$ be the $k$-th pair of basis vectors (parallel to x and y axes, respectively), and $R,C$ 
    be the numbers of grid rows and columns, respectively, thus $1\leqslant i\leqslant R,1\leqslant j\leqslant C$. 
    - If the origin is located at the lower-left corner of the grid, i.e., `loc_ori=='lower_left'`, then the point in 
        the $i$-th row and $j$-th column of the $k$-th grid is
    $$
    p_{ij}^k=(R-i)v_k+(j-1)u_k,
    $$
    - If the original is located at the center of the grid, i.e., `loc_ori=='center'`, then
    $$
    p_{ij}^k=(R-i-\frac{R-1}{2})v_k+(j-1-\frac{C-1}{2})u_k.
    $$
    <figure markdown>
    ![girds](../imgs/grids_from_bases.png){width="480"}
    <figcaption>The point coordinates of the resulting grids with different loc_ori.</figcaption>
    </figure>
    Args:
        uvs: Shape of `[P, 2, 3]`. `uvs[k-1, 0]` represents $u_k$, and `uvs[k-1, 1]` represents $v_k$.
        n_rows: The number of rows of each grid.
        n_cols: The number of columns of each grid.
        loc_ori: `'lower_left'` or `'center'`. The location of the origin for each grid.
    Returns:
        grids: Shape of `[P, R, C, 3]`. `grids[k, i, j]` represents $p_{i+1,j+1}^k$, the point in the $i+1$-th row and 
            $j+1$-th column of the $k$-th grid.
    """
    xs, ys = np.meshgrid(np.arange(n_cols), np.arange(n_rows)) # [R, C] * 2
    ys = np.flip(ys, axis=0)
    xys = np.stack([xs, ys], axis=-1) # [R, C, 2]
    if loc_ori == 'center':
        xys = xys - np.asarray([(n_cols - 1) / 2, (n_rows - 1) / 2])
    elif loc_ori != 'lower_left':
        raise NotImplementedError(f"loc_ori must be 'center' or 'lower_left', but got {loc_ori}")
    xys = xys[..., None] # [R, C, 2, 1]
    uvs = np.asarray(uvs)[:, None, None] # [P, 1, 1, 2, 3]
    grids = np.sum(xys * uvs, axis=-2) #[P, R, C, 3]
    return grids


def grids_polar_from_bases(uvs: ArrayLike, n_angles: int = 120, n_points_dia: int = 100):
    r"""
    Generate polar grids of points from 2D basis vector pairs.
    Specifically, let $u_k,v_k$ be the $k$-th pair of basis vectors (parallel to x and y axes, respectively), 
    $A$ be the numbers of angles/diameters, and $D$ be the number of points on each diameter. The origin is located at 
    the center of the grid, then the point in the $i$-th row and $j$-th column of the $k$-th grid is
    <figure markdown>
    ![girds](../imgs/grids_polar_from_bases.png){width="250"}
    <figcaption>The point coordinates of the resulting grids.</figcaption>
    </figure>
    Args:
        uvs: Shape of `[P, 2, 3]`. `uvs[k-1, 0]` represents $u_k$, and `uvs[k-1, 1]` represents $v_k$.
        n_angles: The number of angles/diameters.
        n_points_dia: The number of points on each diameter.
    Returns:
        grids: Shape of `[P, D, A, 3]`. `grids[k, i, j]` represents $p_{i+1,j+1}^k$, the point in the $i+1$-th row and 
            $j+1$-th column of the $k$-th grid.
    """
    dias = np.arange(n_points_dia) # [D]
    dias = dias - (n_points_dia - 1) / 2
    thetas = np.linspace(0, 2 * np.pi, n_angles) # [A]
    thetas, dias = np.meshgrid(thetas, dias) # [D, A] * 2
    # do not need to flip dias along axis 0, because xs should increase
    xs, ys = dias * np.cos(thetas), dias * np.sin(thetas) # [D, A] * 2
    xys = np.stack([xs, ys], axis=-1) # [D, A, 2]
    xys = xys[..., None] # [D, A, 2, 1]
    uvs = np.asarray(uvs)[:, None, None] # [P, 1, 1, 2, 3]
    grids = np.sum(xys * uvs, axis=-2) #[P, D, A, 3]
    return grids


def sample_array(array: np.ndarray, points: ArrayLike, mode_interp: str = 'linear', value_pad: float = 0):
    r"""
    Sample/Interpolate values of an array at given points (locations). 
    Usually, the array represents an image, and the points represents a multi-dimensional grid of coordinates.
    Specifically, an image $I:\Omega\rightarrow\mathbb{R}^C$ is regarded as a mapping (function) from the image space 
    $\Omega\subset\mathbb{R}^D$ to the space of pixel values, that is, $D=2$ or $3$ for a 2D or 3D image, and $C=1$ or 
    $3$ for a grayscale or RGB image. Therefore, the input array contains sampled values of the image on a regular grid, 
    and the input points contains $D$-dimensional coordinates where the pixel values need to be interpolated.
    Args:
        array: Multi-dimensional. Typically, the shape is `[B, X, Y]` or `[B, X, Y, Z]` for a batch of 2D or 3D 
            grayscale image; `[B, 3, X, Y]` or `[B, 3, X, Y, Z]` for a batch of 2D or 3D RGB images. The dimension of 
            `B` is not needed if only one image is interpolated. Note that for grayscale images, there should not be a 
            dimension with size 1 before `X`, due to Scipy's behavior (see the 
            [Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html) 
            of Scipy for more information).
        points: Shape of `[..., n_dims]`, where `n_dims` is the dimension of `array`. For example, if shapes of `array`
            and 'points' are `[B, X, Y]` and `[5, 7, 3]`, respectively, then `points[i, j]==[b=2, 0.3, 0.4]` means that 
            the point $p_{ij}$ is in the 0.3-th row and 0.4-th column of the $b=2$-nd image.
        mode_interp: `'linear'` or `'nearest'`. The mode of interpolation.
        value_pad: Interpolated value for points outside the array space $\Omega$ (domain, coverage).
    Returns:
        values: Shape of `[...]`. Interpolated values of `array` at `points`.
    Examples:
        >>> from aicafe import img
        >>> img0 = [[0, 1],[2, 3]] # [2, 2]
        >>> img1 = [[4, 5],[6, 7]] # [2, 2]
        >>> array = np.stack([img0, img1], axis=0) # [B, X, Y] == [2, 2, 2]
        >>> # points: shape of [4, 3], where 3 == n_dims
        >>> points = [[0, 0, 1], # the first 0: interpolate img0, the second 0 and 1: interpolate at location [0, 1]
        ...           [0, 0.25, 0.5],
        ...           [1, 1, 1],
        ...           [1, 0.5, 0.25]]
        >>> img.sample_array(array, points)
        >>> array([1.  , 1.  , 7.  , 5.25]) 
    """
    xys = [np.arange(size) for size in array.shape]
    interpolator = RegularGridInterpolator(xys, array, method=mode_interp, bounds_error=False, fill_value=value_pad)
    values = interpolator(points)
    return values


def points_sur_curve(curve: ArrayLike, box: ArrayLike = None, rad: float = 3):
    r"""Get the grid points surrounding a curve.
    Args:
        curve: Shape of `[P, 3]` ($P\geqslant 2$), an array of 3D coordinates representing the points on the curve. 
            The coordinates must be non-negative.
        box: Shape of `[3]`, the bounding box containing the curve and the surrounding points. Note that the box always
            has a vertex of $(0,0,0)$, thus only 3 (rather than 6) parameters are needed. If `None`, it will be 
            calculated as the smallest box that contains the curve.
        rad: A point is surrounding the curve if and only if there exists a point on the curve such that the distance 
            between the two points is less than or equal to `radius`.
    Returns:
        Shape of `[Q, 3]`, an array of 3D coordinates representing the grid points around the curve.
    """
    if box is None:
        box = np.max(curve, axis=0)
    box = np.asarray([box]) - 1
    curve = np.around(curve)
    grid = np.meshgrid(*[np.arange(-rad, rad + 1)] * 3, indexing='ij')
    grid = np.stack(grid, axis=-1)
    dist = np.sum(grid**2, axis=-1)
    idxs_sur = np.stack(np.where(dist <= rad**2), axis=-1) - rad # [I, 3]
    idxs_sur = curve[:, None] + idxs_sur[None] # [P, I, 3]
    idxs_sur = np.unique(idxs_sur.reshape(-1, 3), axis=0) # [I_new, 3]
    idxs_sur = idxs_sur[np.all((idxs_sur >= 0) * (idxs_sur <= box), axis=-1)]
    return idxs_sur


def points_ori2xsec(points: ArrayLike, curve: ArrayLike):
    """Transform given points from original image space (in which the curve is localized) to the space of stacked 
       cross-sectional images (which are generated from points on the curve).
    Args:
        points: Shape of `[P, 3]`, 3D coordinates representing points in the original image space.
        curve: Shape of `[C, 3]`, 3D coordinates representing the points on the curve.
    Returns:
        Shape of `[P, 3]`, 3D coordinates representing points in the cross-sectional image space, each of which 
            corresponds to a point in `points`.
    """
    tans = vecs_tan(curve, normalized=True) # [C, 3]
    cp = points[:, None] - curve[None] # [P, C, 3]
    dot = np.sign(np.sum(cp * tans[None], axis=-1)) # [P, C]
    dist_cp = np.sum(cp**2, axis=-1) # [P, C]
    idx_seg_p = np.empty((cp.shape[0], )) # [P], index of segment that each point belongs to
    ps = []
    for p in range(cp.shape[0]):
        idx_seg_p_candidate = np.where(dot[p, :-1] * dot[p, 1:] <= 0)
        idx_seg_p_candidate = idx_seg_p_candidate[0]
        if len(idx_seg_p_candidate) == 0:
            idx_seg_p[p] = np.inf
        else:
            idx_seg_p[p] = idx_seg_p_candidate[np.argmin(dist_cp[p, idx_seg_p_candidate])]
            ps.append(p)
    points = points[ps]
    idx_seg_p = idx_seg_p[ps].astype(int)

    uv1 = vecs_basis_xsec(tans=tans[:-1]) # [C-1, 2, 3]
    uv1 = uv1[idx_seg_p] # [P, 2, 3]

    t1 = tans[idx_seg_p] # [P, 3]
    t2 = tans[idx_seg_p + 1] # [P, 3]
    c1 = curve[idx_seg_p] # [P, 3]
    c2 = curve[idx_seg_p + 1] # [P, 3]
    t1c1 = np.sum(t1 * c1, axis=-1) # [P]
    t1c2 = np.sum(t1 * c2, axis=-1) # [P]
    t2c1 = np.sum(t2 * c1, axis=-1) # [P]
    t2c2 = np.sum(t2 * c2, axis=-1) # [P]
    t1p = np.sum(t1 * points, axis=-1) # [P]
    t2p = np.sum(t2 * points, axis=-1) # [P]

    a = -t1c1 + t1c2 + t2c1 - t2c2
    b = t1p - t1c2 - t2p - t2c1 + 2 * t2c2
    c_ = t2p - t2c2
    sqrt_delta = np.sqrt(b**2 - 4 * a * c_)
    lam1, lam2 = (-b + sqrt_delta) / (2 * a), (-b - sqrt_delta) / (2 * a) # [P] * 2
    lams = np.stack([lam1, lam2], axis=-1) # [P, 2]
    idxs_real_root = (lams > -1e-4) & (lams < 1 + 1e-4)
    lams[~idxs_real_root] = np.inf
    lams[idxs_real_root] = np.clip(lams[idxs_real_root], 0, 1)
    lam = np.min(lams, axis=-1) # [P]
    idxs_a_0 = (a == 0)
    lam[idxs_a_0] = -c_[idxs_a_0] / b[idxs_a_0]
    assert np.inf not in lam
    z = 1 - lam + idx_seg_p # [P]
    lam = lam[:, None]

    t = lam * t1 + (1 - lam) * t2 # [P, 3]
    rots_t1_to_t = mats_rot(t1, t) # [P, 3, 3]
    uv = rots_t1_to_t[:, None] @ uv1[..., None] # [P, 1, 3, 3] @ [P, 2, 3, 1] == [P, 2, 3, 1]
    uv = uv[..., 0] # [P, 2, 3]
    c = lam * c1 + (1 - lam) * c2 # [P, 3]
    cp = points - c # [P, 3]
    xy = np.sum(cp[:, None] * uv, axis=-1) # [P, 2]
    return np.concatenate([xy, z[:, None]], axis=-1), ps


def getCSImg(self, pos, norm, hheight=100, hwidth=100, src='o'):
    #cross sectional image from pos and norm
    csimg = np.zeros((2 * hheight, 2 * hwidth))
    for v in range(-hheight, hheight):
        for u in range(-hwidth, hwidth):
            csimg[v + hwidth, u + hheight] = self.getInt(getCSPos(norm, pos, u, v), src)
    return csimg


def cs(self, snake, ptid, hheight=100, hwidth=100, src='o'):
    if type(snake) == int:
        if snake < 0 or snake >= len(self.snakelist):
            raise ValueError('No such snake id')
        else:
            snake = self.snakelist[snake]
    pos = snake[ptid].pos
    norm = snake.getNorm(ptid)
    return self.getCSImg(pos, norm, hheight, hwidth, src)


def csStackRange(self, snake, startid=0, endid=-1, hheight=100, hwidth=100, src='o', img=None):
    if type(snake) == int:
        snake = self.snakelist[snake]
    if endid == -1:
        endid = snake.NP

    curve = np.empty((endid - startid, 4)) # [P, 4]
    for i in range(startid, endid):
        pos = snake[i].pos
        curve[i] = [pos.x, pos.y, pos.z, 1]
    if src in self.posRTMat:
        rtm = np.asarray(self.posRTMat[src])
        assert rtm.shape == (4, 3)
        rtm = np.concatenate([rtm[:3], rtm[3].reshape(3, 1)], axis=-1) # [3, 4]
        assert rtm.shape == (3, 4)
        curve = rtm[None, ...] @ curve[..., None] # [1, 3, 4] @ [P, 4, 1] == [P, 3, 1]
        curve = curve[..., 0] # [P, 3]
    else:
        curve = curve[:, :-1]
    if self.rzratio != 1:
        curve[:, -1] = curve[:, -1] / self.rzratio

    uvs = vecs_basis_xsec(curve) # [P, 2, 3]
    grids = grids_rect_from_bases(uvs, n_rows=2 * hheight, n_cols=2 * hwidth, loc_ori='center') # [P, R, C, 3]
    grids = grids + curve[:, None, None]
    assert len(self.I[src].shape) == 3

    if img is None:
        cs_stack = sample_array(self.I[src], grids) # [P, R, C]
    else:
        cs_stack = sample_array(img, grids) # [P, R, C]
    cs_stack = np.moveaxis(cs_stack, 0, -1) # [R, C, P]
    assert cs_stack.shape[-1] == endid - startid
    return cs_stack


def img_cs2ori(self, cs_stack, snake, startid=0, endid=-1, hheight=100, hwidth=100, src='o', rad=5,
               mode_interp='linear'):
    if type(snake) == int:
        snake = self.snakelist[snake]
    if endid == -1:
        endid = snake.NP

    curve = np.empty((endid - startid, 4)) # [P, 4]
    for i in range(startid, endid):
        pos = snake[i].pos
        curve[i - startid] = [pos.x, pos.y, pos.z, 1]
    if src in self.posRTMat:
        rtm = np.asarray(self.posRTMat[src])
        assert rtm.shape == (4, 3)
        rtm = np.concatenate([rtm[:3], rtm[3].reshape(3, 1)], axis=-1) # [3, 4]
        assert rtm.shape == (3, 4)
        curve = rtm[None, ...] @ curve[..., None] # [1, 3, 4] @ [P, 4, 1] == [P, 3, 1]
        curve = curve[:, 0] # [P, 3]
    else:
        curve = curve[:, :-1]
    if self.rzratio != 1:
        curve[:, -1] = curve[:, -1] / self.rzratio

    points = points_sur_curve(curve, box=self.I[src].shape, rad=rad) # [P, 3]
    xyz, ps = points_ori2xsec(points, curve) # [P, 3]
    points = points[ps].astype(int)
    ijk = np.empty_like(xyz)
    ijk[:, 0] = hheight - 0.5 - xyz[:, 1]
    ijk[:, 1] = hwidth - 0.5 + xyz[:, 0]
    ijk[:, 2] = xyz[:, 2]
    ori = np.zeros_like(self.I[src])
    ori[points[:, 0], points[:, 1], points[:, 2]] = sample_array(cs_stack, ijk, mode_interp=mode_interp)
    return ori


def csStackRange_old(self, snake, startid=0, endid=-1, hheight=100, hwidth=100, src='o'):
    if type(snake) == int:
        snake = self.snakelist[snake]
    if endid == -1:
        endid = snake.NP
    cs_stack = np.zeros((2 * hheight, 2 * hwidth, endid - startid))
    for cptid in range(startid, endid):
        if cptid % 10 == 0:
            print('\rGenerating CS', cptid, '/', endid, end='')
        cs_stack[:, :, cptid - startid] = self.cs(snake, cptid, hheight, hwidth, src)
    return cs_stack


def csStackNei(self, snake, ptid, neih, hheight=100, hwidth=100, src='o'):
    if type(snake) == int:
        snake = self.snakelist[snake]
    cs_stack = np.zeros((2 * hheight, 2 * hwidth, neih * 2 + 1))
    for offid in range(-neih, neih + 1):
        cptid = ptid + offid
        if cptid < 0:
            cptid = 0
        if cptid >= snake.NP:
            cptid = snake.NP - 1
        cs_stack[:, :, offid + neih] = self.cs(snake, cptid, hheight, hwidth, src)
    return cs_stack


def getNormImg(self, norm, pos, height, width, alignaxis='x'):
    norm_img = np.zeros((height, width))
    for u in range(-height // 2, height - height // 2):
        for v in range(-width // 2, width - width // 2):
            norm_img[u + height // 2, v + width // 2] = self.getInt(getNormPos(norm, pos, u, v, alignaxis))
    return norm_img


def normPlane(self, snake, ptid, height=200, width=200, alignaxis='x'):
    if type(snake) == int:
        if snake < 0 or snake >= len(self.snakelist):
            print('No such snake id')
            return
        else:
            snake = self.snakelist[snake]
    if isinstance(snake, Snake):
        norm = snake.getNorm(ptid)
        pos = snake[ptid].pos
    else:
        norm = snake
        pos = ptid
    return self.getNormImg(norm, pos, height, width, alignaxis)