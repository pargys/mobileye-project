import operator
import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    return (pts-pp)/focal


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    return pts*focal + pp


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    t = EM[:3, 3]
    tx, ty, tz = t[0], t[1], t[2]
    foe = np.array([tx/tz, ty/tz])
    return R, foe, tz


def rotate(pts, R):
    # rotate the points - pts using R
    ones = np.ones((len(pts[:, 0]), 1), int)
    # ones = np.ones((3, 1), int)
    rot_mat = np.dot(R, np.hstack([pts, ones]).T)
    c = rot_mat[2]
    return (rot_mat[:2]/c).T


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    x, y = 0, 1
    m = (foe[y] - p[y]) / (foe[x] - p[x])
    n = (p[y]*foe[x] - foe[y]*p[x]) / (foe[x] - p[x])
    dist_list = list(map(lambda pt: abs((m*pt[x] + n - pt[y]) / np.sqrt(pow(m, 2) + 1)), norm_pts_rot))
    min_index, min_dist = min(enumerate(dist_list), key=operator.itemgetter(1))
    return min_index, norm_pts_rot[min_index]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    x, y = 0, 1
    dX = tZ*(foe[x] - p_rot[x]) / (p_curr[x] - p_rot[x])
    dY = tZ*(foe[y] - p_rot[y]) / (p_curr[y] - p_rot[y])

    # diff_x = abs(p_curr[x] - p_rot[x])
    # diff_y = abs(p_curr[y] - p_rot[y])

    diff_x = abs(foe[x]-p_curr[x])
    diff_y = abs(foe[y]-p_curr[y])

    ratio = diff_x/(diff_y + diff_x)
    # ratio = diff_x/diff_y

    # print(ratio)
    return dX*ratio + dY*(1-ratio)


    # min_rat, max_rat = (x, y) if (foe[x]-p_curr[x]) < (foe[y]-p_curr[y]) else (y, x)
    # ratio = (foe[min_rat]-p_curr[x]) / (foe[y]-p_curr[y])
    # print(ratio)
    # return dX*ratio + dY*(1-ratio) if min_rat==x else dY*ratio + dX*(1-ratio)
    # return (dX + dY) / 2