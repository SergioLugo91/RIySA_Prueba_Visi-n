import cv2                 
import numpy as np                  
import cv2.aruco as aruco  


# === Funciones que devuelven las esquinas del marcador en coordenadas del "mundo" (ring) ===
def world_corners_for_marker_near_00(ox, oy, s):
    # Devuelve las 4 esquinas (TL, TR, BR, BL) del marcador colocado cerca de (0,0),
    # desplazado en ox, oy hacia dentro del ring. 's' es el lado del ArUco (metros).
    return np.array([
        [ox,        oy       ],   # TL (Top-Left)
        [ox + s,    oy       ],   # TR (Top-Right)
        [ox + s,    oy + s   ],   # BR (Bottom-Right)
        [ox,        oy + s   ]    # BL (Bottom-Left)
    ], dtype=np.float32)

def world_corners_for_marker_near_WH(ox, oy, s):
    # Idem, pero para un marcador cerca de la esquina opuesta (WIDTH, HEIGHT).
    # Se resta s y los offsets para “entrar” desde los bordes derecho e inferior.
    return np.array([
        [WIDTH - ox - s,  HEIGHT - oy - s],  # TL
        [WIDTH - ox,      HEIGHT - oy - s],  # TR
        [WIDTH - ox,      HEIGHT - oy     ], # BR
        [WIDTH - ox - s,  HEIGHT - oy     ]  # BL
    ], dtype=np.float32)

def img_to_world(Hmat, u, v):
    # Proyecta un punto de imagen (u,v) a coordenadas mundo usando la homografía Hmat.
    p = np.array([u, v, 1.0], dtype=np.float32)  # Coordenadas homogéneas en imagen.
    q = Hmat @ p                                 # Transformación por homografía.
    q /= q[2]                                    # Paso a coordenadas cartesianas.
    return float(q[0]), float(q[1])              # x,y en el plano del ring.

# --- Configuración inicial ---
video = ""       # Si pones una ruta de vídeo, se usará ese archivo; vacío => cámara.
camId = 1        # Índice de cámara (0 suele ser la predeterminada).
target_fps = 30.0
frame_period = 1.0 / target_fps  # Periodo de fotogramas (~0.033 s).

# ==== GEOMETRÍA DEL RING (en metros) ====
WIDTH, HEIGHT = 1.20, 0.80     # Ancho y alto del ring en metros.
marker_len = 0.087             # Lado físico del ArUco en metros.

# IDs de tus dos marcadores (ajústalos a los que imprimas).
ID_A = 0   # Marcador cerca de la esquina (0,0).
ID_B = 1   # Marcador cerca de la esquina (W,H).

# Offsets hacia el interior desde cada esquina (metros).
# Convención 2D: +x a la derecha, +y hacia abajo.
ox_A, oy_A = 0.05, 0.05
ox_B, oy_B = 0.05, 0.05

# --- Crear el diccionario y los parámetros del detector ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detectorParams = cv2.aruco.DetectorParameters()   # Parámetros por defecto.
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)  # Detector moderno (OpenCV >= 4.7).

# --- Abrir el video o la cámara ---
inputVideo = cv2.VideoCapture(video if video else camId, cv2.CAP_DSHOW)  # CAP_DSHOW: backend DirectShow en Windows.

# --- CARGAR MATRICES DE CALIBRACIÓN ---
data = np.load("Calibracion/cam_calib_data.npz")     # Carga archivo .npz con K y D.

camMatrix = data["K"].astype(np.float32)             # Matriz intrínseca K (3x3).
distCoeffs = data["D"].astype(np.float32)            # Coefs. de distorsión.

print("Matriz de calibración cargada:")
print(camMatrix)
print("Coeficientes de distorsión:")
print(distCoeffs.ravel())

while True:  # Bucle principal de captura y procesamiento.
    ok, frame = inputVideo.read()     # Lee un frame de la cámara/vídeo.
    if not ok:
        break                         # Si falla la lectura, sal del bucle.

    frame_und = cv2.undistort(frame, camMatrix, distCoeffs)  # Corrige la distorsión.
    gray = cv2.cvtColor(frame_und, cv2.COLOR_BGR2GRAY)       # Pasa a escala de grises.
    corners, ids, _ = detector.detectMarkers(gray)           # Detecta ArUcos.

    if ids is not None:
        # === (1) Dibuja marcadores e IDs detectados ===
        aruco.drawDetectedMarkers(frame_und, corners, ids)   # Dibuja contornos/IDs.
        for i, mid in enumerate(ids.flatten()):
            pts = corners[i][0]              # Esquinas del i-ésimo marcador (4x2).
            center = pts.mean(axis=0)        # Centro del marcador (media de esquinas).
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(frame_und, (cx, cy), 4, (0, 255, 0), -1)  # Punto central.
            cv2.putText(frame_und, f"ID {int(mid)}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        Hmat = None                # Homografía imagen->mundo (se calculará si hay datos).
        ids_flat = ids.flatten()   # Aplana los IDs detectados.
        img_pts = []               # Lista de puntos en imagen (u,v) de esquinas.
        world_pts = []             # Lista de puntos correspondientes en el mundo (x,y).

        # Por cada marcador de interés, añadimos sus 4 esquinas (imagen y mundo).
        for i, mid in enumerate(ids_flat):
            if mid == ID_A:
                img_pts.extend(corners[i][0])  # 4 esquinas del ArUco A (en imagen).
                world_pts.extend(world_corners_for_marker_near_00(ox_A, oy_A, marker_len))
            elif mid == ID_B:
                img_pts.extend(corners[i][0])  # 4 esquinas del ArUco B (en imagen).
                world_pts.extend(world_corners_for_marker_near_WH(ox_B, oy_B, marker_len))

        # Si tenemos al menos 4 correspondencias (1 marcador) podemos estimar una homografía,
        # con 8 puntos (2 marcadores) será más robusta (RANSAC).
        if len(img_pts) >= 4:
            img_pts = np.array(img_pts, dtype=np.float32)
            world_pts = np.array(world_pts, dtype=np.float32)
            Hmat, maskH = cv2.findHomography(
                img_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )

            if Hmat is not None:
                H_inv = np.linalg.inv(Hmat)  # Inversa: mundo->imagen (para dibujar el borde del ring).

                # Proyecta el rectángulo del ring (0,0)-(W,H) desde mundo a imagen para dibujarlo.
                ring_world = np.array([[0,0,1],[WIDTH,0,1],[WIDTH,HEIGHT,1],[0,HEIGHT,1]], dtype=np.float32).T
                ring_img = (H_inv @ ring_world); ring_img /= ring_img[2]
                poly = ring_img[:2].T.reshape(-1,1,2).astype(np.int32)
                cv2.polylines(frame_und, [poly], True, (0,255,255), 2)  # contorno del ring en la imagen

                # Toma el centro de la imagen en píxeles y lo lleva a coordenadas del mundo.
                h_, w_ = frame_und.shape[:2]
                xw, yw = img_to_world(Hmat, w_/2, h_/2)

                # Comprueba si ese punto mundo cae dentro del rectángulo del ring.
                inside = (0 <= xw <= WIDTH) and (0 <= yw <= HEIGHT)
                cv2.putText(
                    frame_und,
                    f"x={xw:.3f}m  y={yw:.3f}m  {'DENTRO' if inside else 'FUERA'}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if inside else (0,0,255), 2
                )

    # === (3) Mostrar resultado y esperar tecla ===
    cv2.imshow("Deteccion ArUco (Ring)", frame_und)   # Muestra el frame con anotaciones.
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir.
        break

inputVideo.release()         # Libera la cámara/vídeo.
cv2.destroyAllWindows()      # Cierra ventanas de OpenCV.

