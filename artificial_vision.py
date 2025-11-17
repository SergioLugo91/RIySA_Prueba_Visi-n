import cv2
import numpy as np
import cv2.aruco as aruco
from collections import deque
import time
import math
import socket
from dataclasses import dataclass


# ============================================================================
# MAPEO DE ARUCOS A ROBOTS
# ============================================================================
# Diccionario que mapea ArUcos individuales a IDs de robots
# Formato: {aruco_id: robot_id}
# TODO: Rellenar con los IDs reales de los ArUcos
ARUCO_TO_ROBOT = {
    98: 1,   # ArUco 98 pertenece al Robot 1
    124: 1,  # ArUco 124 pertenece al Robot 1
    62: 2,   # ArUco 62 pertenece al Robot 2
    203: 2,  # ArUco 203 pertenece al Robot 2
    6: 3,    # ArUco 6 pertenece al Robot 3
    7: 3,    # ArUco 7 pertenece al Robot 3
    # Añadir más robots según sea necesario
}

def get_robot_id_from_aruco(aruco_id):
    """
    Obtiene el ID del robot a partir del ID del ArUco.
    
    Args:
        aruco_id: ID del marcador ArUco
    
    Returns:
        int: ID del robot, o None si no está mapeado
    """
    return ARUCO_TO_ROBOT.get(aruco_id, None)


# ============================================================================
# CLASE PARA FILTRO DE MEDIANA DE ESQUINAS DEL RING
# ============================================================================
class MedianCornerFilter:
    """
    Clase que almacena las últimas N muestras de las esquinas del cuadrado
    y devuelve la mediana de cada coordenada para suavizar el tintineo.
    Mantiene el último valor conocido cuando no hay detección.
    """
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.buffer = deque(maxlen=num_samples)
        self.last_known_corners = None
        self.frames_without_detection = 0
        self.max_frames_without_detection = 30  # ~1 segundo a 30 fps
    
    def add_sample(self, corners):
        if corners is not None and len(corners) == 4:
            self.buffer.append(corners.copy())
            self.frames_without_detection = 0
    
    def get_median_corners(self):
        if len(self.buffer) > 0:
            samples = np.array(list(self.buffer))
            median_corners = np.median(samples, axis=0).astype(np.float32)
            self.last_known_corners = median_corners.copy()
            return median_corners
        else:
            self.frames_without_detection += 1
            if self.frames_without_detection > self.max_frames_without_detection:
                self.last_known_corners = None
            return self.last_known_corners
    
    def is_ready(self):
        return len(self.buffer) == self.num_samples
    
    def is_using_cached_value(self):
        return len(self.buffer) == 0 and self.last_known_corners is not None
    
    def reset(self):
        self.buffer.clear()
        self.last_known_corners = None
        self.frames_without_detection = 0


# ============================================================================
# CLASE DE DATOS UBOT
# ============================================================================
@dataclass
class Ubot:
    id: int         # ID del ROBOT (no del ArUco)
    ang: float
    dist: float
    Out: int


# ============================================================================
# FUNCIONES AUXILIARES PARA RING
# ============================================================================
def world_corners_for_marker_near_00(ox, oy, s):
    """Esquinas del marcador cerca de (0,0) en coordenadas del mundo"""
    return np.array([
        [-ox - s, -oy - s],  # TL
        [-ox,     -oy - s],  # TR
        [-ox,     -oy    ],  # BR
        [-ox - s, -oy    ]   # BL
    ], dtype=np.float32)


def world_corners_for_marker_near_WH(ox, oy, s, width, height):
    """Esquinas del marcador cerca de (WIDTH, HEIGHT) en coordenadas del mundo"""
    return np.array([
        [width + ox,     height + oy    ],  # TL
        [width + ox + s, height + oy    ],  # TR
        [width + ox + s, height + oy + s],  # BR
        [width + ox,     height + oy + s]   # BL
    ], dtype=np.float32)


def img_to_world(Hmat, u, v):
    """Proyecta un punto de imagen (u,v) a coordenadas mundo usando homografía"""
    p = np.array([u, v, 1.0], dtype=np.float32)
    q = Hmat @ p
    q /= q[2]
    return float(q[0]), float(q[1])


# ============================================================================
# FUNCIONES AUXILIARES PARA PARES DE ARUCOS
# ============================================================================
def rotationMatrixToEulerAngles(R):
    """Convierte matriz de rotación a ángulos de Euler (roll, pitch, yaw)"""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])


def getRTMatrix(R_, T_, forceType=-1):
    """Obtiene matriz de transformación 4x4 desde rvec y tvec"""
    M = None
    R = R_.copy()
    T = T_.copy()
    
    if R.dtype == np.float64:
        Matrix = np.eye(4, dtype=np.float64)
        R33 = Matrix[0:3, 0:3]
        
        if R.size == 3:
            R33[:, :] = cv2.Rodrigues(R)[0]
        elif R.size == 9:
            R64 = R.astype(np.float64)
            R33[:, :] = R64.reshape(3, 3)
        
        for i in range(3):
            Matrix[i, 3] = T.flat[i] if T.ndim > 1 else T[i]
        M = Matrix
        
    elif R.dtype == np.float32:
        Matrix = np.eye(4, dtype=np.float32)
        R33 = Matrix[0:3, 0:3]
        
        if R.size == 3:
            R33[:, :] = cv2.Rodrigues(R)[0]
        elif R.size == 9:
            R32 = R.astype(np.float32)
            R33[:, :] = R32.reshape(3, 3)
        
        for i in range(3):
            Matrix[i, 3] = T.flat[i] if T.ndim > 1 else T[i]
        M = Matrix
    
    if forceType == -1:
        return M
    else:
        return M.astype(forceType)


def calculate_distance_between_markers(tvec1, tvec2):
    """Calcula la distancia 3D entre dos marcadores"""
    pos1 = tvec1.flatten()
    pos2 = tvec2.flatten()
    return np.linalg.norm(pos1 - pos2)


def normalize_angle_deg(a):
    """Normaliza ángulo a rango [-180, 180]"""
    return ((a + 180) % 360) - 180


def safe_norm(v):
    """Normaliza vector evitando división por cero"""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v * 0.0


# ============================================================================
# CONFIGURACIÓN WIFI
# ============================================================================
IP = "255.255.255.255"
PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("", PORT))


def enviarDatos(sock, msg, ip, port):
    sock.sendto(msg.encode(), (ip, port))
    print(f"[Enviado -> {ip}:{port}] {msg}")


# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================
video = ""
camId = 1
target_fps = 30.0
frame_period = 1.0 / target_fps

# --- Geometría del ring (en metros) ---
WIDTH, HEIGHT = 0.80, 0.80
ring_marker_len = 0.09

# IDs de marcadores del ring
ID_RING_A = 0  # Marcador cerca de (0, 0)
ID_RING_B = 1  # Marcador cerca de (WIDTH, HEIGHT)

# Offsets de marcadores del ring
ox_A, oy_A = 0.0, 0.03
ox_B, oy_B = 0.0, 0.05

# --- Geometría de marcadores de robots (en metros) ---
robot_marker_len = 0.025

# --- Parámetros de detección ---
estimatePose = True
showRejected = True

# --- Crear diccionario y detector ArUco ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

# --- Inicializar filtro de mediana para el ring ---
corner_filter = MedianCornerFilter(num_samples=10)

# --- Abrir cámara ---
inputVideo = cv2.VideoCapture(camId, cv2.CAP_DSHOW)

# --- Cargar calibración ---
data = np.load("Calibracion/cam_calib_data.npz")
camMatrix = data["K"].astype(np.float32)
distCoeffs = data["D"].astype(np.float32)

print("Matriz de calibración cargada:")
print(camMatrix)
print("Coeficientes de distorsión:")
print(distCoeffs.ravel())

# --- Sistema de coordenadas del marcador (para robots) ---
objPoints_robot = np.array([
    [-robot_marker_len/2,  robot_marker_len/2, 0],
    [ robot_marker_len/2,  robot_marker_len/2, 0],
    [ robot_marker_len/2, -robot_marker_len/2, 0],
    [-robot_marker_len/2, -robot_marker_len/2, 0]
], dtype=np.float32)

# --- Control de pares ---
current_pair_index = 0


# ============================================================================
# BUCLE PRINCIPAL
# ============================================================================
while True:
    loop_start = time.time()
    
    ok, frame = inputVideo.read()
    if not ok:
        break
    
    # --- Corrección de distorsión ---
    frame_und = cv2.undistort(frame, camMatrix, distCoeffs)
    gray = cv2.cvtColor(frame_und, cv2.COLOR_BGR2GRAY)
    
    # --- Detectar marcadores ---
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Diccionario para almacenar datos de marcadores
    marker_data = {}  # {id: {"center": (x, y), "rvec": rvec, "tvec": tvec, "corners": c, "robot_id": robot_id}}
    
    # ========================================================================
    # PROCESAMIENTO DE MARCADORES DETECTADOS
    # ========================================================================
    both_ring_markers_detected = False
    
    if ids is not None:
        # Dibujar marcadores detectados
        aruco.drawDetectedMarkers(frame_und, corners, ids)
        
        # Procesar cada marcador
        for i, c in enumerate(corners):
            marker_id = int(ids[i][0])
            pts = c[0]
            center = pts.mean(axis=0)
            center_x, center_y = int(center[0]), int(center[1])
            
            # Obtener ID del robot asociado
            robot_id = get_robot_id_from_aruco(marker_id)
            
            # Dibujar centro y ID
            cv2.circle(frame_und, (center_x, center_y), 4, (0, 255, 0), -1)
            
            # Mostrar ID de ArUco y Robot
            if robot_id is not None:
                cv2.putText(frame_und, f"ArUco {marker_id} (R{robot_id})", 
                           (center_x + 8, center_y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame_und, f"ID {marker_id}", 
                           (center_x + 8, center_y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Estimar pose (solo para robots, no para marcadores del ring)
            if marker_id > 1:
                success, rvec, tvec = cv2.solvePnP(objPoints_robot, c, camMatrix, distCoeffs)
                if success:
                    # Guardar datos del marcador
                    marker_data[marker_id] = {
                        "center": (center_x, center_y),
                        "rvec": rvec,
                        "tvec": tvec,
                        "corners": c,
                        "robot_id": robot_id
                    }
                    
                    # Dibujar ejes
                    cv2.drawFrameAxes(frame_und, camMatrix, distCoeffs, rvec, tvec, robot_marker_len * 1.5)
        
        # ====================================================================
        # PROCESAMIENTO DEL RING (IDs 0 y 1)
        # ====================================================================
        ids_flat = ids.flatten()
        has_marker_A = ID_RING_A in ids_flat
        has_marker_B = ID_RING_B in ids_flat
        both_ring_markers_detected = has_marker_A and has_marker_B
        
        if both_ring_markers_detected:
            img_pts = []
            world_pts = []
            
            for i, mid in enumerate(ids_flat):
                if mid == ID_RING_A:
                    img_pts.extend(corners[i][0])
                    world_pts.extend(world_corners_for_marker_near_00(ox_A, oy_A, ring_marker_len))
                elif mid == ID_RING_B:
                    img_pts.extend(corners[i][0])
                    world_pts.extend(world_corners_for_marker_near_WH(ox_B, oy_B, ring_marker_len, WIDTH, HEIGHT))
            
            if len(img_pts) >= 8:
                img_pts = np.array(img_pts, dtype=np.float32)
                world_pts = np.array(world_pts, dtype=np.float32)
                Hmat, maskH = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                
                if Hmat is not None:
                    H_inv = np.linalg.inv(Hmat)
                    
                    # Proyectar ring a imagen
                    ring_world = np.array([[0, 0, 1], [WIDTH, 0, 1], [WIDTH, HEIGHT, 1], [0, HEIGHT, 1]], dtype=np.float32).T
                    ring_img = (H_inv @ ring_world)
                    ring_img /= ring_img[2]
                    ring_corners_raw = ring_img[:2].T.astype(np.float32)
                    
                    # Añadir al filtro de mediana
                    corner_filter.add_sample(ring_corners_raw)
                    
                    # Verificar si centro de imagen está dentro del ring
                    h_, w_ = frame_und.shape[:2]
                    xw, yw = img_to_world(Hmat, w_/2, h_/2)
                    inside = (0 <= xw <= WIDTH) and (0 <= yw <= HEIGHT)
                    
                    cv2.putText(frame_und, f"Centro: x={xw:.3f}m  y={yw:.3f}m  {'DENTRO' if inside else 'FUERA'}",
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if inside else (0, 0, 255), 2)
        else:
            # Advertencia de marcadores del ring faltantes
            missing = []
            if not has_marker_A:
                missing.append(f"ID {ID_RING_A}")
            if not has_marker_B:
                missing.append(f"ID {ID_RING_B}")
            
            if len(missing) > 0:
                cv2.putText(frame_und, f"Falta marcador ring: {', '.join(missing)}",
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # ========================================================================
    # DIBUJAR RING
    # ========================================================================
    ring_corners_median = corner_filter.get_median_corners()
    
    if ring_corners_median is not None:
        poly = ring_corners_median.reshape(-1, 1, 2).astype(np.int32)
        
        if corner_filter.is_using_cached_value():
            cv2.polylines(frame_und, [poly], True, (0, 0, 255), 2)
            cv2.putText(frame_und, f"RING CACHE - {corner_filter.frames_without_detection} frames",
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif corner_filter.is_ready():
            cv2.polylines(frame_und, [poly], True, (0, 255, 0), 2)
        else:
            cv2.polylines(frame_und, [poly], True, (0, 255, 255), 2)
            cv2.putText(frame_und, f"Ring inicializando: {len(corner_filter.buffer)}/{corner_filter.num_samples}",
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # ========================================================================
    # PROCESAMIENTO DE ROBOTS - CALCULAR DISTANCIAS ENTRE DIFERENTES ROBOTS
    # ========================================================================
    # Agrupar marcadores por robot (solo necesitamos uno por robot)
    robots_with_markers = {}  # {robot_id: aruco_id}
    
    for aruco_id, data in marker_data.items():
        robot_id = data["robot_id"]
        if robot_id is not None:
            # Guardar solo el primer ArUco detectado de cada robot
            if robot_id not in robots_with_markers:
                robots_with_markers[robot_id] = aruco_id
    
    # Lista de robots detectados
    detected_robot_ids = sorted(robots_with_markers.keys())
    num_robots = len(detected_robot_ids)
    
    if num_robots >= 2:
        # Mostrar información
        cv2.putText(frame_und, f"Robots detectados: {num_robots}",
                   (20, frame_und.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Usar el primer robot como referencia
        base_robot_id = detected_robot_ids[0]
        base_aruco_id = robots_with_markers[base_robot_id]
        
        rvecBase = marker_data[base_aruco_id]["rvec"]
        tvecBase = marker_data[base_aruco_id]["tvec"]
        M_base = getRTMatrix(rvecBase, tvecBase)
        M_base_inv = np.linalg.inv(M_base)
        
        # CALCULAR DISTANCIAS Y ÁNGULOS ENTRE TODOS LOS PARES DE ROBOTS
        robot_pairs_data = []
        
        for i in range(num_robots):
            robot_A_id = detected_robot_ids[i]
            aruco_A_id = robots_with_markers[robot_A_id]
            
            for j in range(i + 1, num_robots):
                robot_B_id = detected_robot_ids[j]
                aruco_B_id = robots_with_markers[robot_B_id]
                
                # Obtener datos de los ArUcos
                rvecA = marker_data[aruco_A_id]["rvec"]
                tvecA = marker_data[aruco_A_id]["tvec"]
                rvecB = marker_data[aruco_B_id]["rvec"]
                tvecB = marker_data[aruco_B_id]["tvec"]
                
                # Construir matrices RT 4x4
                M_A = getRTMatrix(rvecA, tvecA)
                M_B = getRTMatrix(rvecB, tvecB)
                
                # Transformar al sistema del marcador base
                M_A_in_base = M_base_inv @ M_A
                M_B_in_base = M_base_inv @ M_B
                
                # Extraer vectores de traslación
                tA_in_base = M_A_in_base[0:3, 3].reshape(3, 1)
                tB_in_base = M_B_in_base[0:3, 3].reshape(3, 1)
                
                # Posiciones
                pos_A = M_A_in_base[0:3, 3]
                pos_B = M_B_in_base[0:3, 3]
                v_AB = pos_B - pos_A
                
                # Normales (ejes Z)
                n_A = M_A_in_base[0:3, 2]
                n_B = M_B_in_base[0:3, 2]
                
                # Proyecciones en plano XY
                nA_xy = safe_norm(np.array([n_A[0], n_A[1], 0.0]))
                nB_xy = safe_norm(np.array([n_B[0], n_B[1], 0.0]))
                vAB_xy = safe_norm(np.array([v_AB[0], v_AB[1], 0.0]))
                vBA_xy = -vAB_xy
                
                # Ángulos
                az_nA = math.atan2(nA_xy[1], nA_xy[0])
                az_vAB = math.atan2(vAB_xy[1], vAB_xy[0])
                az_nB = math.atan2(nB_xy[1], nB_xy[0])
                az_vBA = math.atan2(vBA_xy[1], vBA_xy[0])
                
                delta_A = normalize_angle_deg(math.degrees(az_vAB - az_nA))
                delta_B = normalize_angle_deg(math.degrees(az_vBA - az_nB))
                
                # Distancia
                distancia = calculate_distance_between_markers(tA_in_base, tB_in_base)
                
                # Redondear
                delta_A = round(delta_A, 3)
                delta_B = round(delta_B, 3)
                distancia = round(distancia, 3)
                
                # Enviar datos al ROBOT A sobre el ROBOT B
                ubot_A = Ubot(id=robot_A_id, ang=float(delta_A), dist=float(distancia), Out=robot_B_id)
                enviarDatos(sock, str(ubot_A), IP, PORT)
                
                # Enviar datos al ROBOT B sobre el ROBOT A
                ubot_B = Ubot(id=robot_B_id, ang=float(delta_B), dist=float(distancia), Out=robot_A_id)
                enviarDatos(sock, str(ubot_B), IP, PORT)
                
                # Guardar para visualización
                robot_pairs_data.append({
                    'robot_A_id': robot_A_id,
                    'robot_B_id': robot_B_id,
                    'aruco_A_id': aruco_A_id,
                    'aruco_B_id': aruco_B_id,
                    'delta_A': delta_A,
                    'delta_B': delta_B,
                    'distancia': distancia,
                    'pA': marker_data[aruco_A_id]["center"],
                    'pB': marker_data[aruco_B_id]["center"]
                })
                
                print(f"[Robot {robot_A_id} -> Robot {robot_B_id}] ángulo={delta_A:.1f}°, dist={distancia:.3f}m")
                print(f"[Robot {robot_B_id} -> Robot {robot_A_id}] ángulo={delta_B:.1f}°, dist={distancia:.3f}m")
        
        # DIBUJAR EL PAR SELECCIONADO
        if len(robot_pairs_data) > 0:
            current_pair_index = min(current_pair_index, len(robot_pairs_data) - 1)
            pair_data = robot_pairs_data[current_pair_index]
            
            pA = pair_data['pA']
            pB = pair_data['pB']
            mid = ((pA[0] + pB[0])//2, (pA[1] + pB[1])//2)
            
            cv2.line(frame_und, pA, pB, (0, 0, 255), 2)
            cv2.putText(frame_und, f"R{pair_data['robot_A_id']}->R{pair_data['robot_B_id']}: {pair_data['delta_A']:.1f}deg",
                       (mid[0] - 100, mid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame_und, f"Dist: {pair_data['distancia']*100:.1f}cm",
                       (mid[0] - 70, mid[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame_und, f"R{pair_data['robot_B_id']}->R{pair_data['robot_A_id']}: {pair_data['delta_B']:.1f}deg",
                       (mid[0] - 100, mid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.putText(frame_und, f"Par: {current_pair_index + 1}/{len(robot_pairs_data)} (A/D)",
                       (20, frame_und.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    elif num_robots == 1:
        cv2.putText(frame_und, f"Solo 1 robot detectado (necesitas 2+)",
                   (20, frame_und.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # ========================================================================
    # MOSTRAR RESULTADO
    # ========================================================================
    cv2.imshow("Vision Artificial - Ring + Robots", frame_und)
    
    # Control de teclado
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break
    elif key == 97:  # 'a' - par anterior
        current_pair_index = max(0, current_pair_index - 1)
    elif key == 100:  # 'd' - par siguiente
        # Calcular número de pares disponibles
        temp_robots = {}
        for aruco_id, data in marker_data.items():
            robot_id = data["robot_id"]
            if robot_id is not None and robot_id not in temp_robots:
                temp_robots[robot_id] = aruco_id
        
        num_temp_robots = len(temp_robots)
        num_pairs = max(0, (num_temp_robots * (num_temp_robots - 1)) // 2)
        
        if num_pairs > 0:
            current_pair_index = min(num_pairs - 1, current_pair_index + 1)

# ============================================================================
# LIMPIEZA
# ============================================================================
inputVideo.release()
cv2.destroyAllWindows()
sock.close()