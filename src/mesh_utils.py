import numpy as np

epsilon = 10 ** -5


#  size        = lambda v : np.sqrt(np.sum(np.dot(v,v)))   TODO: delete all commented shit
#   zero_clip   = lambda x : max(x,epsilon)
# points_diff = lambda p1,p2 : [p1[i] - p2[i] for i in range(len(p1))]
# points_dist = lambda p1,p2 : np.sqrt(np.sum(np.square(points_diff(p1,p2))))  ------ points_distance
# rad_to_deg  = lambda rad: 360*rad/(2*np.pi)       TODO: chagne with math.degrees in handler function_for_constaint


def get_size(v):
    dot = np.dot(v, v)
    s = np.sum(dot)
    return np.sqrt(s)


def get_angle(v1, v2):
    s1 = get_size(v1)
    s2 = get_size(v2)
    dot = np.dot(v1, v2) / max((s1 * s2), epsilon)
    assert type(dot) == np.float64, f'Angle result types: dot={type(dot)}\b       v1={type(v1)}\n       v2={type(v2)}'
    a = np.arccos(dot)
    return a


def points_distance(p1, p2):
    d = [p1[i] - p2[i] for i in range(len(p1))]
    temp = np.square(d)
    s = sum(temp)
    return np.sqrt(s)


# calculate point's min distance from a given plane
def min_dis_p2plane(point, plane):
    a, b, c, d = plane
    x, y, z = point
    temp = abs(a * x + b * y + c * z + d)
    s = get_size(np.array((a, b, c)))
    if (s != 0):
        res = temp / s
    else:
        res = np.inf
    return res
