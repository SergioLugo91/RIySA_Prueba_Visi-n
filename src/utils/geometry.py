# Contenido del archivo src/utils/geometry.py

import numpy as np

def world_corners_for_marker_near_00(ox, oy, s):
    return np.array([
        [-ox - s, -oy - s],  # TL (Top-Left)
        [ox, -oy - s],       # TR (Top-Right)
        [ox, -oy],           # BR (Bottom-Right)
        [ox - s, -oy]        # BL (Bottom-Left)
    ], dtype=np.float32)

def world_corners_for_marker_near_WH(ox, oy, s, width, height):
    return np.array([
        [width + ox, height + oy],       # TL
        [width + ox + s, height + oy],   # TR
        [width + ox + s, height + oy + s], # BR
        [width + ox, height + oy + s]     # BL
    ], dtype=np.float32)

def img_to_world(Hmat, u, v):
    p = np.array([u, v, 1.0], dtype=np.float32)
    q = Hmat @ p
    q /= q[2]
    return float(q[0]), float(q[1])