import numpy as np
from PIL import  Image, ImageOps
import Laba1 as gc

f = open("model_1.obj")

vectorv = []
vectorf = []
for line in f:
    v = line.split()

    if (v[0] == "v"):
        vectorv.append([float(v[1]), float(v[2]), float(v[3])])
    if (v[0] == "f"):
        v1 = v[1].split('/')[0]
        v2 = v[2].split('/')[0]
        v3 = v[3].split('/')[0]
        vectorf.append([int(v1), int(v2), int(v3)])
        img_mat2 = np.zeros(shape=(gc.high, gc.widh, 3), dtype=np.uint8)
z_buff_mat = np.full((gc.high, gc.widh), np.inf, dtype=np.float64)
vectorv = gc.resize(vectorv, 0, -0.05, 0.5 , np.radians(0), np.radians(0), np.radians(0))

for i in range(0,len(vectorf)):
    x0 = ((vectorv[vectorf[i][0] - 1][0]))
    y0 = ((vectorv[vectorf[i][0] - 1][1]))
    z0 = ((vectorv[vectorf[i][0] - 1][2]))
    x1 = ((vectorv[vectorf[i][1] - 1][0]))
    y1 = ((vectorv[vectorf[i][1] - 1][1]))
    z1 = ((vectorv[vectorf[i][1] - 1][2]))
    x2 = ((vectorv[vectorf[i][2] - 1][0]))
    y2 = ((vectorv[vectorf[i][2] - 1][1]))
    z2 = ((vectorv[vectorf[i][2] - 1][2]))
    color = (200 * gc.sekator_nelic_gran(x0, y0, z0, x1, y1, z1, x2, y2, z2), 100, 200)
    if (gc.sekator_nelic_gran(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0):
        gc.draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, img_mat2, color, z_buff_mat)


img = Image.fromarray(img_mat2, mode="RGB")
img = ImageOps.flip(img)
img.save("img.jpg")