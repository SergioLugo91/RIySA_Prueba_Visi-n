# Transformaciones de coordenadas entre diferentes sistemas

import numpy as np

def img_to_world(Hmat, u, v):
    p = np.array([u, v, 1.0], dtype=np.float32)  # Coordenadas homogéneas en imagen.
    q = Hmat @ p                                 # Transformación por homografía.
    q /= q[2]                                    # Paso a coordenadas cartesianas.
    return float(q[0]), float(q[1])              # x,y en el plano del mundo.

def world_corners_for_marker_near_00(ox, oy, s):
    return np.array([
        [-ox - s, -oy - s],  # TL (Top-Left)
        [ox, -oy - s],       # TR (Top-Right)
        [ox, -oy],           # BR (Bottom-Right)
        [ox - s, -oy]        # BL (Bottom-Left)
    ], dtype=np.float32)

def world_corners_for_marker_near_WH(ox, oy, s, width, height):
    return np.array([
        [width + ox, height + oy],      # TL
        [width + ox + s, height + oy],  # TR
        [width + ox + s, height + oy + s],  # BR
        [width + ox, height + oy + s]   # BL
    ], dtype=np.float32)