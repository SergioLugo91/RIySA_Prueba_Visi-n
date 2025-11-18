import cv2
import numpy as np
import math
import serial
from dataclasses import dataclass
import socket



# --- Configuración WIFI ---
IP = "255.255.255.255"
PORT = 8888

datos_robots = []

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("", PORT))

def enviarDatos(sock, msg, ip, port):
    sock.sendto(msg.encode(), (ip, port))
    print(f"[Enviado -> {ip}:{port}] {msg}")

# --- Configuración inicial ---
video = ""       # ruta de video, o dejar vacío para usar cámara
camId = 0        # ID de cámara (0 = cámara predeterminada)
markerLength = 0.025  # longitud del marcador en metros
estimatePose = True
showRejected = True
target_fps = 30.0
frame_period = 1.0 / target_fps  # ~0.033 segundos
#ser = serial.Serial()
#ser.baudrate = 19200
#ser.port = 'COM1'

# --- Crear el diccionario y los parámetros del detector ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

# --- Abrir el video o la cámara ---
if video:
    inputVideo = cv2.VideoCapture(video)
    waitTime = 0
else:
    inputVideo = cv2.VideoCapture(camId)
    waitTime = 10

# --- CARGAR MATRICES DE CALIBRACIÓN ---
data = np.load("Calibracion/cam_calib_data.npz")

camMatrix = data["K"].astype(np.float32)
distCoeffs = data["D"].astype(np.float32)

print("Matriz de calibración cargada:")
print(camMatrix)
print("Coeficientes de distorsión:")
print(distCoeffs.ravel())

# --- Definir el sistema de coordenadas del marcador ---
objPoints = np.array([
    [-markerLength/2,  markerLength/2, 0],
    [ markerLength/2,  markerLength/2, 0],
    [ markerLength/2, -markerLength/2, 0],
    [-markerLength/2, -markerLength/2, 0]
], dtype=np.float32)

# --- Función auxiliar: matriz de rotación → ángulos de Euler ---
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])  # roll, pitch, yaw

# --- Función auxiliar: obtener matriz RT 4x4 ---
def getRTMatrix(R_, T_, forceType=-1):
    M = None
    R = R_.copy()
    T = T_.copy()
    
    if R.dtype == np.float64:
        assert T.dtype == np.float64
        Matrix = np.eye(4, dtype=np.float64)
        R33 = Matrix[0:3, 0:3]
        
        if R.size == 3:
            R33[:,:] = cv2.Rodrigues(R)[0]
        elif R.size == 9:
            R64 = R.astype(np.float64)
            R33[:,:] = R64.reshape(3, 3)
        
        for i in range(3):
            Matrix[i, 3] = T.flat[i] if T.ndim > 1 else T[i]
        M = Matrix
        
    elif R.dtype == np.float32:
        Matrix = np.eye(4, dtype=np.float32)
        R33 = Matrix[0:3, 0:3]
        
        if R.size == 3:
            R33[:,:] = cv2.Rodrigues(R)[0]
        elif R.size == 9:
            R32 = R.astype(np.float32)
            R33[:,:] = R32.reshape(3, 3)
        
        for i in range(3):
            Matrix[i, 3] = T.flat[i] if T.ndim > 1 else T[i]
        M = Matrix
    
    if forceType == -1:
        return M
    else:
        MTyped = M.astype(forceType)
        return MTyped
    

# --- Función para calcular distancia 3D entre dos marcadores ---
def calculate_distance_between_markers(tvec1, tvec2):
    """ Calcula la distancia 3D en metros entre dos marcadores usando sus poses """
    # Convertir los vectores de traslación a coordenadas 3D
    pos1 = tvec1.flatten()
    pos2 = tvec2.flatten()
    
    # Calcular distancia euclidiana en metros
    distance = np.linalg.norm(pos1 - pos2)
    return distance

# --- Función para normalizar ángulo a [-180, 180] grados ---
def normalize_angle_deg(a):
    return ((a + 180) % 360) - 180

# --- Control de pares ---
current_pair_index = 0

# --- Clase de datos Ubot ---
@dataclass
class Ubot:
    id: int
    ang: float
    dist: float
    Out: int

# --- Clase ArucoRobotDetector ---
class ArucoRobotDetector:
    """Detector de robots usando pares de ArUcos"""
    
    def __init__(self, robot_markers=None, marker_length=0.025,
                 calibration_path="Calibracion/cam_calib_data.npz"):
        # Configuración por defecto
        if robot_markers is None:
            self.robot_markers = {
                1: [2, 3],
                2: [4, 5],
                3: [6, 7],
                4: [8, 9]
            }
        else:
            self.robot_markers = robot_markers
        
        self.marker_length = marker_length
        
        # Detector ArUco
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)
        
        # Calibración
        data = np.load(calibration_path)
        self.cam_matrix = data["K"].astype(np.float32)
        self.dist_coeffs = data["D"].astype(np.float32)
        
        # Puntos 3D del marcador
        self.obj_points = np.array([
            [-marker_length/2, marker_length/2, 0],
            [marker_length/2, marker_length/2, 0],
            [marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)
        
        print("ArucoRobotDetector: Calibración cargada")
    
    @staticmethod
    def calculate_distance(tvec1, tvec2):
        """Calcula distancia 3D entre dos posiciones"""
        return np.linalg.norm(tvec1.flatten() - tvec2.flatten())
    
    @staticmethod
    def calculate_angle(tvec1, tvec2):
        """Calcula ángulo relativo entre dos posiciones"""
        delta = tvec2.flatten() - tvec1.flatten()
        return math.degrees(math.atan2(delta[1], delta[0]))
    
    def process_frame(self, frame):
        """
        Procesa un frame y detecta robots.
        
        Returns:
            dict: {
                'frame': frame procesado,
                'markers': lista de marcadores detectados,
                'robots': {robot_id: {'center': (x,y), 'tvec': tvec, 'marker_ids': [id1, id2]}},
                'pairs': [{'r1': id, 'r2': id, 'dist': float, 'ang': float}]
            }
        """
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        result = {
            'frame': frame.copy(),
            'markers': [],
            'robots': {},
            'pairs': []
        }
        
        if ids is None:
            return result
        
        # Dibujar marcadores
        cv2.aruco.drawDetectedMarkers(result['frame'], corners, ids)
        
        # Detectar cada marcador y calcular pose
        markers_dict = {}
        for i, c in enumerate(corners):
            marker_id = int(ids[i][0])
            center = c[0].mean(axis=0)
            
            success, rvec, tvec = cv2.solvePnP(self.obj_points, c, 
                                               self.cam_matrix, self.dist_coeffs)
            if success:
                markers_dict[marker_id] = {
                    'corners': c,
                    'center': tuple(center.astype(int)),
                    'rvec': rvec,
                    'tvec': tvec
                }
                
                # Dibujar ejes
                cv2.drawFrameAxes(result['frame'], self.cam_matrix, self.dist_coeffs,
                                 rvec, tvec, self.marker_length * 1.5)
        
        result['markers'] = list(markers_dict.keys())
        
        # Procesar robots (cada robot tiene 2 marcadores)
        for robot_id, marker_ids in self.robot_markers.items():
            if all(mid in markers_dict for mid in marker_ids):
                m1 = markers_dict[marker_ids[0]]
                m2 = markers_dict[marker_ids[1]]
                
                # Centro promedio
                cx = int((m1['center'][0] + m2['center'][0]) / 2)
                cy = int((m1['center'][1] + m2['center'][1]) / 2)
                tvec_avg = (m1['tvec'] + m2['tvec']) / 2
                
                result['robots'][robot_id] = {
                    'center': (cx, cy),
                    'tvec': tvec_avg,
                    'marker_ids': marker_ids
                }
                
                # Dibujar robot
                cv2.circle(result['frame'], (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(result['frame'], f"R{robot_id}", (cx + 15, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calcular distancias y ángulos entre robots
        robot_ids = list(result['robots'].keys())
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                r1 = robot_ids[i]
                r2 = robot_ids[j]
                
                tvec1 = result['robots'][r1]['tvec']
                tvec2 = result['robots'][r2]['tvec']
                
                dist = self.calculate_distance(tvec1, tvec2)
                ang = self.calculate_angle(tvec1, tvec2)
                
                result['pairs'].append({
                    'r1': r1,
                    'r2': r2,
                    'dist': dist,
                    'ang': ang
                })
                
                # Dibujar línea entre robots
                pt1 = result['robots'][r1]['center']
                pt2 = result['robots'][r2]['center']
                cv2.line(result['frame'], pt1, pt2, (255, 255, 0), 2)
                
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                cv2.putText(result['frame'], f"D:{dist*100:.1f}cm A:{ang:.1f}°",
                           (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return result


# Si se ejecuta directamente
if __name__ == "__main__":
    detector = ArucoRobotDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.process_frame(frame)
        cv2.imshow("Robot Detector", result['frame'])
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# --- Bucle principal ---
while True:
    loop_start = time.time()

    ret, image = inputVideo.read()
    if not ret:
        break
    
    imageCopy = image.copy()
    corners, ids, rejected = detector.detectMarkers(image)

    marker_data = {}  # {id: {"center": (x, y), "rvec": rvec, "tvec": tvec, "corners": c}}

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(imageCopy, corners, ids)

        for i, c in enumerate(corners):
            marker_id = int(ids[i][0])
            center_x = int(np.mean(c[0][:, 0]))
            center_y = int(np.mean(c[0][:, 1]))

            success, rvec, tvec = cv2.solvePnP(objPoints, c, camMatrix, distCoeffs)
            if not success:
                continue

            # Guardar datos
            marker_data[marker_id] = {
                "center": (center_x, center_y),
                "rvec": rvec,
                "tvec": tvec,
                "corners": c
            }

            # Dibujar ejes y yaw en la imagen
            cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, markerLength * 1.5)

    # --- Agrupar en pares excluyendo el marcador base ---
    sorted_ids = sorted(marker_data.keys())

    # --- Identificar marcador base (el de ID más bajo) ---
    id_base = None
    if len(sorted_ids) != 0:
        id_base = sorted_ids[0]
        pBase = marker_data[id_base]["center"]
        rvecBase = marker_data[id_base]["rvec"]
        tvecBase = marker_data[id_base]["tvec"]

    # --- Preparar pares ---
    others = sorted_ids[1:]  # excluye el base
    paired_ids = [others[i:i+2] for i in range(0, len(others), 2)]
    num_pairs = len(paired_ids)

    # --- Dibujar solo el par actual ---
    if id_base is not None and 0 <= current_pair_index < num_pairs:
        pair = paired_ids[current_pair_index]
        if len(pair) == 2:
            idA, idB = pair
            # --- Obtener datos de los marcadores A y B ---
            rvecA = marker_data[idA]["rvec"]
            tvecA = marker_data[idA]["tvec"]
            rvecB = marker_data[idB]["rvec"]
            tvecB = marker_data[idB]["tvec"]

            # --- Construir matrices RT 4x4 (marker -> camera) ---
            M_base = getRTMatrix(rvecBase, tvecBase)   # base -> camera
            M_A = getRTMatrix(rvecA, tvecA)            # A -> camera
            M_B = getRTMatrix(rvecB, tvecB)            # B -> camera

            # --- Transformar las poses de A y B al sistema del marcador base ---
            # Queremos M_{X->base} = inv(M_base) @ M_X
            M_base_inv = np.linalg.inv(M_base)

            M_A_in_base = M_base_inv @ M_A
            M_B_in_base = M_base_inv @ M_B

            # --- Extraer vectores de traslación (3x1) en sistema base ---
            tA_in_base = M_A_in_base[0:3, 3].reshape(3, 1)
            tB_in_base = M_B_in_base[0:3, 3].reshape(3, 1)

            # --- Extraer matrices de rotación (3x3) en sistema base ---
            RA_in_base = M_A_in_base[0:3, 0:3]
            RB_in_base = M_B_in_base[0:3, 0:3]

            # --- Convertir matrices de rotación a ángulos de Euler ---
            roll_A, pitch_A, yaw_A = rotationMatrixToEulerAngles(RA_in_base)
            roll_B, pitch_B, yaw_B = rotationMatrixToEulerAngles(RB_in_base)

            # --- Posiciones de A y B en coordenadas del marcador base ---
            pos_A = M_A_in_base[0:3, 3]
            pos_B = M_B_in_base[0:3, 3]
            v_AB = pos_B - pos_A  # vector A -> B

            n_A = M_A_in_base[0:3, 2]  # eje +Z local (normal) de A en sistema base
            n_B = M_B_in_base[0:3, 2]  # eje +Z local (normal) de B en sistema base

            # --- Proyecciones de las normales y del vector de conexión en el plano XY ---
            nA_xy = np.array([n_A[0], n_A[1], 0.0])
            nB_xy = np.array([n_B[0], n_B[1], 0.0])
            vAB_xy = np.array([v_AB[0], v_AB[1], 0.0])
            vBA_xy = -vAB_xy

            # --- Normalizar vectores y evitar divisiones por cero ---
            def safe_norm(v):
                n = np.linalg.norm(v)
                return v / n if n > 1e-8 else v * 0.0

            nA_xy = safe_norm(nA_xy)
            nB_xy = safe_norm(nB_xy)
            vAB_xy = safe_norm(vAB_xy)
            vBA_xy = safe_norm(vBA_xy)

            # --- Ángulos (azimut) de cada vector ---
            az_nA = math.atan2(nA_xy[1], nA_xy[0])
            az_vAB = math.atan2(vAB_xy[1], vAB_xy[0])
            az_nB = math.atan2(nB_xy[1], nB_xy[0])
            az_vBA = math.atan2(vBA_xy[1], vBA_xy[0])

            # --- Diferencia angular en grados (cuánto girar en yaw global) ---
            delta_A = normalize_angle_deg(math.degrees(az_vAB - az_nA))
            delta_B = normalize_angle_deg(math.degrees(az_vBA - az_nB))

            # --- Distancia entre A y B en el sistema base ---
            distancia = calculate_distance_between_markers(tA_in_base, tB_in_base)

            # --- DEBUG: imprimir resultados ---
            print("--------------------------------------------------")
            print(f"[PAIR {idA}-{idB}]")
            print(f"Posición A (en base): {pos_A}")
            print(f"Posición B (en base): {pos_B}")
            print(f"Ángulo de A hacia B (en su plano): {delta_A:.2f}°")
            print(f"Ángulo de B hacia A (en su plano): {delta_B:.2f}°")
            print(f"Distancia: {distancia:.3f} m")
            print("--------------------------------------------------")

            # --- Crear objetos Ubot ---
            delta_A = round(delta_A,3)
            delta_B = round(delta_B,3)
            distancia = round(distancia,3)
            ubot_pair_A = Ubot(id=idA, ang=float(delta_A), dist=float(distancia), Out=0)
            enviarDatos(sock, str(ubot_pair_A), IP, PORT)
            ubot_pair_B = Ubot(id=idB, ang=float(delta_B), dist=float(distancia), Out=0)
            enviarDatos(sock, str(ubot_pair_B), IP, PORT)
            
            # --- Dibujar información en la imagen ---
            pA = marker_data[idA]["center"]
            pB = marker_data[idB]["center"]
            mid = ((pA[0]+pB[0])//2, (pA[1]+pB[1])//2)

            cv2.line(imageCopy, pA, pB, (0, 0, 255), 2)
            cv2.putText(imageCopy, f"{idA}->{idB}: {delta_A:.1f} deg", (mid[0]-70, mid[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(imageCopy, f"Dist:{distancia*100:.1f} cm", (mid[0]-70, mid[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(imageCopy, f"{idB}->{idA}: {delta_B:.1f} deg", (mid[0]-70, mid[1]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(imageCopy, f"Dist:{distancia*100:.1f} cm", (mid[0]-70, mid[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            print(f"[PAIR {idA}-{idB}] delta_A = {delta_A:.1f}°, delta_B = {delta_B:.1f}°, dist = {distancia:.3f} m")

    cv2.imshow("Control de Pares (A / D)", imageCopy)

    # --- Control de teclado ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 97:  # a
        current_pair_index = max(0, current_pair_index - 1)
    elif key == 100:  # d
        current_pair_index = min(num_pairs - 1, current_pair_index + 1)
    
    # --- Control de frecuencia (30 Hz) ---
    elapsed = time.time() - loop_start
    sleep_time = frame_period - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

inputVideo.release()
cv2.destroyAllWindows()
