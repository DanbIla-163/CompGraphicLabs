import numpy as np
from math import sin, cos


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quaternion_rotate_vector(q, v):
    v = np.array([0, *v])
    q_conj = quaternion_conjugate(q)
    v_rotated = quaternion_multiply(q, quaternion_multiply(v, q_conj))
    return v_rotated[1:]  # Возвращаем только векторную часть


def euler_to_quaternion(alpha, betta, gamma):
    cy = cos(gamma * 0.5)
    sy = sin(gamma * 0.5)
    cp = cos(betta * 0.5)
    sp = sin(betta * 0.5)
    cr = cos(alpha * 0.5)
    sr = sin(alpha * 0.5)

    w = cr * cp * cy + sr * cp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def transform_mesh(vectorv, translation=(0, 0, 0), rotation=None, euler_angles=None, scale=1.0):


    if rotation is None and euler_angles is not None:
        rotation = euler_to_quaternion(*euler_angles)

    if rotation is not None:
        rotation = rotation / np.linalg.norm(rotation)

    if np.isscalar(scale):
        scale = np.array([scale, scale, scale])
    else:
        scale = np.array(scale)

    transformed_vectorv = []
    for vec in vectorv:
        v_scaled = np.array(vec) * scale

        if rotation is not None:
            v_rotated = quaternion_rotate_vector(rotation, v_scaled)
        else:
            v_rotated = v_scaled

        v_translated = v_rotated + np.array(translation)

        transformed_vectorv.append(v_translated.tolist())

    return transformed_vectorv