# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Image (only gray-scale image, i.e., matrix) transformation utilities.

References
----------
- Leptonica: Rotation
  http://www.leptonica.com/rotation.html
- Image rotation by MATLAB without using imrotate
  https://stackoverflow.com/a/19687481/4856091
  https://stackoverflow.com/a/19689081/4856091
- Stackoverflow: Python: Rotating greyscale images
  https://codereview.stackexchange.com/a/41903
"""


import numpy as np
import numba as nb
from scipy import ndimage


@nb.jit([nb.float64[:, :](nb.int64[:, :], nb.float64, nb.boolean,
                          nb.boolean, nb.float64),
         nb.float64[:, :](nb.float64[:, :], nb.float64, nb.boolean,
                          nb.boolean, nb.float64)],
        nopython=True)
def rotate_center(imgin, angle, interp=True, reshape=True, fill_value=0.0):
    """
    Rotate the input image (only gray-scale image currently supported)
    by a given angle about its center.

    Parameters
    ----------
    imgin : 2D `~numpy.ndarray`
        Input gray-scale image to be rotated
    angle : float
        Rotation angle (unit: [ degree ])
    interp : bool, optional
        Use the area mapping of the 4 closest input pixels (``interp=True``),
        which is also the same as "bilinear interpolation",
        or use the nearest neighbor (``interp=False``) for rotated pixels.
    reshape : bool, optional
        Whether adapt the output shape so that the input image is contained
        completely in the output?
    fill_value : float, optional
        Value used to fill pixels in the rotated image that corresponding
        pixels outside the boundaries of the input image.
    """
    nrow, ncol = imgin.shape
    # Rotation transformation image
    angle = np.deg2rad(angle)
    mrotate = np.zeros((2, 2), dtype=np.float64)
    mrotate[0, 0] = np.cos(angle)
    mrotate[0, 1] = np.sin(angle)
    mrotate[1, 0] = -np.sin(angle)
    mrotate[1, 1] = np.cos(angle)
    # Determine the shape of rotated image
    corner00 = np.array((0, 0))
    corner01 = np.array((0, ncol-1))
    corner10 = np.array((nrow-1, 0))
    corner11 = np.array((nrow-1, ncol-1))
    corners = np.vstack((corner00, corner01, corner10, corner11))
    if reshape:
        dest = np.dot(corners.astype(np.float64), mrotate)
        # XXX: ``numba`` does not support ``np.max()`` with arguments
        minr = np.min(dest[:, 0])
        minc = np.min(dest[:, 1])
        maxr = np.max(dest[:, 0])
        maxc = np.max(dest[:, 1])
        nr = int(maxr - minr + 0.5)
        nc = int(maxc - minc + 0.5)
    else:
        # Constraint to be same shape
        nr = nrow
        nc = ncol
    imgout = np.ones((nr, nc)) * fill_value
    #
    # Calculate the offset, for easier transformation of rotated pixels
    # back to input image.
    #
    # NOTE:
    # Notations:
    #     P_out : (r_out, c_out) a pixel in the output rotated image
    #     C_out : center position of the output rotated image
    #     P_in : (r_in, c_in) a pixel in the input image
    #     C_in : center position of the input image
    #     R : rotation matrix
    #     R_T : transposed rotation matrix
    # The rotation relation between pixel pair is (about the center):
    #     (P_in - C_in) * R = P_out - C_out
    # Then:
    #     (P_in - C_in) = (P_out - C_out) * R_T
    # And then:
    #     P_in = C_in + (P_out-C_out) * R_T = P_out*R_T + (C_in - C_out*R_T)
    # Thus can define the "offset" as:
    #     offset = C_in - C_out * R_T
    # Then the transformation back to input image is simply given by:
    #     P_in = P_out * R_T + offset
    #
    center_in = np.array((nrow/2.0 - 0.5, ncol/2.0 - 0.5))
    center_out = np.array((nr/2.0 - 0.5, nc/2.0 - 0.5))
    mrotate_T = mrotate.transpose()
    offset = center_in - np.dot(center_out, mrotate_T)
    # Map pixels of the rotated image to the input image
    for rr in range(nr):
        for cc in range(nc):
            p_out = np.array((rr, cc))
            p_in = np.dot(p_out.astype(np.float64), mrotate_T) + offset
            if np.all((p_in > corner00) & (p_in < corner11)):
                # Calculate the pixel value for the rotated image
                if interp:
                    # Use area mapping of the 4 closest input pixels
                    idx_rf, idx_cf = np.floor(p_in).astype(np.int64)
                    idx_rc, idx_cc = np.ceil(p_in).astype(np.int64)
                    # NOTE:
                    # It is possible that ``p_in[0]`` and/or ``p_in[1]``
                    # are just integers, which cause ``idx_rf == idx_rc``
                    # and/or ``idx_cf == idx_cc``, which further lead to
                    # the calculated pixel value ``p_val = 0``.
                    if idx_rf == idx_rc:
                        idx_rc += 1
                    if idx_cf == idx_cc:
                        idx_cc += 1
                    # Calculate the overlapping areas
                    p_r, p_c = p_in
                    p4_area = np.array([(idx_rc - p_r) * (idx_cc - p_c),
                                        (idx_rc - p_r) * (p_c - idx_cf),
                                        (p_r - idx_rf) * (idx_cc - p_c),
                                        (p_r - idx_rf) * (p_c - idx_cf)])
                    p4_val = np.array((imgin[idx_rf, idx_cf],
                                       imgin[idx_rf, idx_cc],
                                       imgin[idx_rc, idx_cf],
                                       imgin[idx_rc, idx_cc]))
                    p_val = np.sum(p4_area * p4_val)
                else:
                    # Use the nearest neighbor as the rotated value
                    idx_r = round(p_in[0])
                    idx_c = round(p_in[1])
                    p_val = imgin[idx_r, idx_c]
                #
                imgout[rr, cc] = p_val
    return imgout


def circle2ellipse(imgcirc, bfraction, rotation=0.0):
    """
    Shrink the input circle image with respect to the center along the
    column (axis) to transform the circle to an ellipse, and then rotate
    around the image center.

    Parameters
    ----------
    imgcirc : 2D `~numpy.ndarray`
        Input image grid containing a circle at the center
    bfraction : float
        The fraction of the semi-minor axis w.r.t. the semi-major axis
        (i.e., the half width of the input image), to determine the
        shrunk size (height) of the output image.
        Should be a fraction within [0, 1]
    rotation : float, optional
        Rotation angle (unit: [ degree ])

    Returns
    -------
    imgout : 2D `~numpy.ndarray`
        Image of the same size as the input circle image.
    """
    nrow, ncol = imgcirc.shape
    # Shrink the circle to be elliptical
    nrow2 = nrow * bfraction
    nrow2 = int(nrow2 / 2) * 2 + 1  # be odd
    # NOTE: zoom() calculate the output shape with round() instead of int();
    #       fix the warning about they may be different.
    zoom = ((nrow2+0.1)/nrow, 1)
    img2 = ndimage.zoom(imgcirc, zoom=zoom, order=1)
    # Pad the shrunk image to have the same size as input
    imgout = np.zeros(shape=(nrow, ncol))
    r1 = int((nrow - nrow2) / 2)
    imgout[r1:(r1+nrow2), :] = img2
    # Rotate the ellipse
    imgout = ndimage.rotate(imgout, angle=rotation, reshape=False, order=1)
    return imgout
