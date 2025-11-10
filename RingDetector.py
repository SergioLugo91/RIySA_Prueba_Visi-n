import cv2                 
import numpy as np                  
import cv2.aruco as aruco  
from collections import deque


# === Clase para calcular la mediana de las esquinas del cuadrado ===
class MedianCornerFilter:
    """
    Clase que almacena las últimas N muestras de las esquinas del cuadrado
    y devuelve la mediana de cada coordenada para suavizar el tintineo.
    Mantiene el último valor conocido cuando no hay detección.
    """
    def __init__(self, num_samples=10):
        """
        Args:
            num_samples (int): Número de muestras a almacenar para el cálculo de la mediana.
        """
        self.num_samples = num_samples
        # Buffer circular para almacenar las esquinas (cada elemento es un array de 4 puntos)
        self.buffer = deque(maxlen=num_samples)
        # Almacena el último valor conocido de las esquinas
        self.last_known_corners = None
        # Contador de frames sin detección
        self.frames_without_detection = 0
        # Máximo de frames sin detección antes de resetear (opcional)
        self.max_frames_without_detection = 30  # ~1 segundo a 30 fps
    
    def add_sample(self, corners):
        """
        Añade una nueva muestra de esquinas al buffer.
        
        Args:
            corners (np.ndarray): Array de 4 puntos (esquinas) con forma (4, 2).
                                  Orden: [TL, TR, BR, BL]
        """
        if corners is not None and len(corners) == 4:
            self.buffer.append(corners.copy())
            self.frames_without_detection = 0  # Resetea el contador
    
    def get_median_corners(self):
        """
        Calcula la mediana de todas las muestras almacenadas.
        Si no hay muestras nuevas, devuelve el último valor conocido.
        
        Returns:
            np.ndarray: Array de 4 puntos (esquinas) con la mediana calculada,
                        o el último valor conocido, o None si nunca hubo detección.
        """
        if len(self.buffer) > 0:
            # Convierte el buffer a un array 3D: (num_muestras, 4_esquinas, 2_coords)
            samples = np.array(list(self.buffer))
            
            # Calcula la mediana a lo largo del eje de las muestras (eje 0)
            median_corners = np.median(samples, axis=0).astype(np.float32)
            
            # Actualiza el último valor conocido
            self.last_known_corners = median_corners.copy()
            
            return median_corners
        else:
            # No hay muestras nuevas, devuelve el último valor conocido
            self.frames_without_detection += 1
            
            # Opcionalmente, resetea si pasa mucho tiempo sin detección
            if self.frames_without_detection > self.max_frames_without_detection:
                self.last_known_corners = None
                
            return self.last_known_corners
    
    def is_ready(self):
        """
        Verifica si el buffer está lleno y listo para usar.
        
        Returns:
            bool: True si el buffer tiene el número completo de muestras.
        """
        return len(self.buffer) == self.num_samples
    
    def is_using_cached_value(self):
        """
        Verifica si se está usando un valor en caché (sin detección nueva).
        
        Returns:
            bool: True si no hay detección nueva y se usa el último valor conocido.
        """
        return len(self.buffer) == 0 and self.last_known_corners is not None
    
    def reset(self):
        """Limpia el buffer de muestras y el último valor conocido."""
        self.buffer.clear()
        self.last_known_corners = None
        self.frames_without_detection = 0


# === Funciones que devuelven las esquinas del marcador en coordenadas del "mundo" (ring) ===
def world_corners_for_marker_near_00(ox, oy, s):
    # Devuelve las 4 esquinas (TL, TR, BR, BL) del marcador colocado cerca de (0,0),
    # desplazado en ox, oy hacia dentro del ring. 's' es el lado del ArUco (metros).
    return np.array([
        [-ox -s,  -oy - s ],   # TL (Top-Left)
        [ox,      -oy - s ],   # TR (Top-Right)
        [ox,      -oy     ],   # BR (Bottom-Right)
        [ox - s,  -oy     ]    # BL (Bottom-Left)
    ], dtype=np.float32)

def world_corners_for_marker_near_WH(ox, oy, s):
    # Idem, pero para un marcador FUERA de la esquina opuesta (WIDTH, HEIGHT).
    # Se suma s y los offsets para "salir" desde los bordes derecho e inferior.
    return np.array([
        [WIDTH + ox,      HEIGHT + oy     ],  # TL
        [WIDTH + ox + s,  HEIGHT + oy     ],  # TR
        [WIDTH + ox + s,  HEIGHT + oy + s ],  # BR
        [WIDTH + ox,      HEIGHT + oy + s ]   # BL
    ], dtype=np.float32)

def img_to_world(Hmat, u, v):
    # Proyecta un punto de imagen (u,v) a coordenadas mundo usando la homografía Hmat.
    p = np.array([u, v, 1.0], dtype=np.float32)  # Coordenadas homogéneas en imagen.
    q = Hmat @ p                                 # Transformación por homografía.
    q /= q[2]                                    # Paso a coordenadas cartesianas.
    return float(q[0]), float(q[1])              # x,y en el plano del ring.

# --- Configuración inicial ---
video = ""       # Si pones una ruta de vídeo, se usará ese archivo; vacío => cámara.
camId = 1       # Índice de cámara (0 suele ser la predeterminada).
target_fps = 30.0
frame_period = 1.0 / target_fps  # Periodo de fotogramas (~0.033 s).

# ==== GEOMETRÍA DEL RING (en metros) ====
# WIDTH, HEIGHT = 0.07, 0.15     # Ancho y alto del ring en metros.
# marker_len = 0.045             # Lado físico del ArUco en metros.

# ==== GEOMETRÍA DEL RING (en metros) ====
WIDTH, HEIGHT = 0.80, 0.80     # Ancho y alto del ring en metros.
marker_len = 0.087             # Lado físico del ArUco en metros.

# IDs de tus dos marcadores (ajústalos a los que imprimas).
ID_A = 0   # Marcador cerca de la esquina (0,0).
ID_B = 1   # Marcador cerca de la esquina (W,H).

# Offsets hacia el interior desde cada esquina (metros).
# Convención 2D: +x a la derecha, +y hacia abajo.
ox_A, oy_A = 0.02, 0.02
ox_B, oy_B = 0.02, 0.02

# Offsets hacia el EXTERIOR desde cada esquina (metros).
# Convención 2D: +x a la derecha, +y hacia abajo.
# ox_A, oy_A = 0.005, 0.005  
# ox_B, oy_B = 0.005, 0.005  

# --- Crear el diccionario y los parámetros del detector ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detectorParams = cv2.aruco.DetectorParameters()   # Parámetros por defecto.
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)  # Detector moderno (OpenCV >= 4.7).

# --- Inicializar el filtro de mediana para las esquinas del ring ---
corner_filter = MedianCornerFilter(num_samples=10)

# --- Abrir el video o la cámara ---
inputVideo = cv2.VideoCapture(camId, cv2.CAP_DSHOW)  # CAP_DSHOW: backend DirectShow en Windows.
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

    # Variable para controlar si se detectaron AMBOS marcadores
    both_markers_detected = False
    
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

        # Verificar si ambos marcadores están presentes
        has_marker_A = ID_A in ids_flat
        has_marker_B = ID_B in ids_flat
        both_markers_detected = has_marker_A and has_marker_B

        # Solo procesar si AMBOS marcadores están presentes
        if both_markers_detected:
            # Por cada marcador de interés, añadimos sus 4 esquinas (imagen y mundo).
            for i, mid in enumerate(ids_flat):
                if mid == ID_A:
                    img_pts.extend(corners[i][0])  # 4 esquinas del ArUco A (en imagen).
                    world_pts.extend(world_corners_for_marker_near_00(ox_A, oy_A, marker_len))
                elif mid == ID_B:
                    img_pts.extend(corners[i][0])  # 4 esquinas del ArUco B (en imagen).
                    world_pts.extend(world_corners_for_marker_near_WH(ox_B, oy_B, marker_len))

            # Con 8 puntos (2 marcadores) calculamos la homografía
            if len(img_pts) >= 8:
                img_pts = np.array(img_pts, dtype=np.float32)
                world_pts = np.array(world_pts, dtype=np.float32)
                Hmat, maskH = cv2.findHomography(
                    img_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )

                if Hmat is not None:
                    H_inv = np.linalg.inv(Hmat)  # Inversa: mundo->imagen (para dibujar el borde del ring).

                    # Proyecta el rectángulo del ring (0,0)-(W,H) desde mundo a imagen.
                    ring_world = np.array([[0,0,1],[WIDTH,0,1],[WIDTH,HEIGHT,1],[0,HEIGHT,1]], dtype=np.float32).T
                    ring_img = (H_inv @ ring_world); ring_img /= ring_img[2]
                    ring_corners_raw = ring_img[:2].T.astype(np.float32)  # (4, 2)
                    
                    # Añade la muestra actual al filtro de mediana
                    corner_filter.add_sample(ring_corners_raw)
                    
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
        else:
            # Si falta algún marcador, mostrar advertencia
            missing = []
            if not has_marker_A:
                missing.append(f"ID {ID_A}")
            if not has_marker_B:
                missing.append(f"ID {ID_B}")
            
            cv2.putText(frame_und, f"Falta marcador: {', '.join(missing)}", 
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

    # Obtiene las esquinas suavizadas con la mediana (siempre, incluso si no hay detección nueva)
    ring_corners_median = corner_filter.get_median_corners()
    
    if ring_corners_median is not None:
        # Usa las esquinas suavizadas para dibujar
        poly = ring_corners_median.reshape(-1,1,2).astype(np.int32)
        
        # Dibuja el cuadrado con diferentes colores según el estado
        if corner_filter.is_using_cached_value():
            # Rojo cuando se usa el valor en caché (sin detección nueva)
            cv2.polylines(frame_und, [poly], True, (0,0,255), 2)
            cv2.putText(frame_und, f"CACHE - Sin deteccion: {corner_filter.frames_without_detection} frames",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        elif corner_filter.is_ready():
            # Verde cuando el filtro tiene todas las muestras
            cv2.polylines(frame_und, [poly], True, (0,255,0), 2)
        else:
            # Amarillo durante la fase de inicialización
            cv2.polylines(frame_und, [poly], True, (0,255,255), 2)
            cv2.putText(frame_und, f"Inicializando: {len(corner_filter.buffer)}/{corner_filter.num_samples}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # === (3) Mostrar resultado y esperar tecla ===
    cv2.imshow("Deteccion ArUco (Ring)", frame_und)   # Muestra el frame con anotaciones.
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir.
        break

inputVideo.release()         # Libera la cámara/vídeo.
cv2.destroyAllWindows()      # Cierra ventanas de OpenCV.

