# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Interpolation utilities.
"""


def bilinear(x, y, p11, p12, p21, p22):
    """
    Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

    p11 --+-- p12
     |    |    |
     +--(x,y)--+
     |    |    |
    p21 --+-- p22

    Credit
    ------
    [1] http://en.wikipedia.org/wiki/Bilinear_interpolation
    [2] https://stackoverflow.com/a/8662355/4856091
    """
    # Sort points by X then Y
    points = sorted([p11, p12, p21, p22])
    x1, y1, q11 = points[0]
    _x1, y2, q12 = points[1]
    x2, _y1, q21 = points[2]
    _x2, _y2, q22 = points[3]

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError("points do not form a rectangle")
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError("(x, y) not within the rectangle")
    a1 = (q11 * (x2 - x) * (y2 - y) + q21 * (x - x1) * (y2 - y) +
          q12 * (x2 - x) * (y - y1) + q22 * (x - x1) * (y - y1))
    a2 = (x2 - x1) * (y2 - y1)
    q = a1 / a2
    return q
