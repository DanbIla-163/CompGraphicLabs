import math
import numpy as np

widh = 2400
high = 1350
coef = 0.2

def dotted_line(image, x0, y0, x1, y1, color):
    count = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):
    xchange = False

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in np.arange(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

def bresanham(image, x0, y0, x1, y1, color):
    xchange = False

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1-y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in np.arange(x0, x1):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if derror > (x1 - x0):
            derror -= 2*(x1-x0)
            y += y_update

def baricenter (x0, y0, x1, y1, x2, y2, x, y):

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def draw_triangle(img_mat, z_buffer, color, polygon_texture_nums, texture_coords, texture, x0, y0, z0, x1, y1, z1, x2, y2, z2,  i0, i1, i2):
    a = 6000
    px0, py0 = a * x0 / z0 + widh / 2, a * y0 / z0 + high / 2
    px1, py1 = a * x1 / z1 + widh / 2, a * y1 / z1 + high / 2
    px2, py2 = a * x2 / z2 + widh / 2, a * y2 / z2 + high / 2

    I0 = i0[2]
    I1 = i1[2]
    I2 = i2[2]

    xmin = min(px0, px1, px2)
    xmax = max(px0, px1, px2)
    ymin = min(py0, py1, py2)
    ymax = max(py0, py1, py2)

    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > widh): xmax = widh
    if (ymax > high): ymax = high

    xmin = math.floor(xmin)
    ymin = math.floor(ymin)
    xmax = math.ceil(xmax)
    ymax = math.ceil(ymax)

    for x in range (xmin, xmax):
        for y in range (ymin, ymax):
            lam0, lam1, lam2 = baricenter(px0, py0, px1, py1, px2, py2, x, y)
            if (lam0 >= 0 and lam1>= 0 and lam2 >= 0):
                z_coord = lam0*z0 + lam1*z1 + lam2*z2
                if z_coord > z_buffer[y][x]:
                    continue
                else:
                    Intence = lam0 * I0 + lam1 * I1 + lam2 * I2
                    UVt1 = polygon_texture_nums[0]
                    UVt2 = polygon_texture_nums[1]
                    UVt3 = polygon_texture_nums[2]

                    Wid = texture.shape[0]
                    High = texture.shape[1]

                    color = texture[int(Wid * (
                                lam0 * texture_coords[UVt1 - 1][1] + lam1 * texture_coords[UVt2 - 1][1] + lam2 *
                                texture_coords[UVt3 - 1][1]))][int(High * (
                                lam0 * texture_coords[UVt1 - 1][0] + lam1 * texture_coords[UVt2 - 1][0] + lam2 *
                                texture_coords[UVt3 - 1][0]))]
                    img_mat[y][x] = color * -Intence
                    z_buffer[y][x] = z_coord

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])

def sekator_nelic_gran(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    l = [0,0,1]
    n = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    return np.dot(n, l) / (np.linalg.norm(n) * np.linalg.norm(l))


def model_rotation(alpha=np.radians(0), beta=np.radians(0), gamma=np.radians(0)):
    Rx = np.array(
        [[1, 0, 0],
         [0, np.cos(alpha), np.sin(alpha)],
         [0, -np.sin(alpha), np.cos(alpha)]]
    )
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)],
         [0, 1, 0],
         [-np.sin(beta), 0, np.cos(beta)]]
    )
    Rz = np.array(
        [[np.cos(gamma), np.sin(gamma), 0],
         [-np.sin(gamma), np.cos(gamma), 0],
         [0, 0, 1]]
    )
    return np.dot(Rx, np.dot(Ry, Rz))


def resize(vectorv, tx=0, ty=0, tz=0, alpha=np.radians(0), beta=np.radians(0), gamma=np.radians(0)):
    for i in vectorv:
        i[0], i[1], i[2] = np.dot(model_rotation(alpha, beta, gamma), [i[0], i[1], i[2]]) + [tx, ty, tz]
    return vectorv


def get_point_normals(vectorv, vectorf, normalsArr):
    for i in range(0, len(vectorf)):
        x0 = (vectorv[vectorf[i][0] - 1][0])
        y0 = (vectorv[vectorf[i][0] - 1][1])
        z0 = (vectorv[vectorf[i][0] - 1][2])
        x1 = (vectorv[vectorf[i][1] - 1][0])
        y1 = (vectorv[vectorf[i][1] - 1][1])
        z1 = (vectorv[vectorf[i][1] - 1][2])
        x2 = (vectorv[vectorf[i][2] - 1][0])
        y2 = (vectorv[vectorf[i][2] - 1][1])
        z2 = (vectorv[vectorf[i][2] - 1][2])

        norm = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        norm /= np.linalg.norm(norm)
        normalsArr[vectorf[i][0] - 1] += norm
        normalsArr[vectorf[i][1] - 1] += norm
        normalsArr[vectorf[i][2] - 1] += norm
    for i in range(0, len(normalsArr)):
        normalsArr[i] = normalsArr[i]/np.linalg.norm(normalsArr[i])

    return normalsArr