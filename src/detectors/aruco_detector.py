import cv2
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class Marker:
    id: int
    corners: np.ndarray
    center: tuple
    rvec: np.ndarray
    tvec: np.ndarray

class ArUcoDetector:
    """
    Detector de marcadores ArUco con funcionalidad completa:
    - Detección básica de marcadores
    - Cálculo de poses 3D
    - Gestión de robots con dos marcadores cada uno
    - Cálculo de distancia y ángulo entre robots
    """
    def __init__(self, marker_length=0.05, cam_id=0, target_fps=30.0,
                 calibration_path="calibracion/cam_calib_data.npz",
                 aruco_dict=cv2.aruco.DICT_6X6_250,
                 robot_markers=None):
        """
        Args:
            marker_length (float): Longitud del lado del marcador en metros
            cam_id (int): ID de la cámara
            target_fps (float): FPS objetivo
            calibration_path (str): Ruta al archivo de calibración
            aruco_dict: Tipo de diccionario ArUco
            robot_markers (dict): Diccionario {robot_id: [marker_id1, marker_id2]}
        """
        self.marker_length = marker_length
        self.cam_id = cam_id
        self.target_fps = target_fps
        self.frame_period = 1.0 / target_fps
        
        # Configuración de marcadores por robot
        # Diccionario por defecto: cada robot tiene 2 marcadores ArUco
        if robot_markers is None:
            self.robot_markers = {
                1: [2, 3],    # Robot 1: ArUco IDs 2 y 3
                2: [4, 5],    # Robot 2: ArUco IDs 4 y 5
                3: [6, 7],    # Robot 3: ArUco IDs 6 y 7
            }
        else:
            self.robot_markers = robot_markers
        
        # Crear diccionario inverso: marker_id -> robot_id
        self.marker_to_robot = {}
        for robot_id, marker_ids in self.robot_markers.items():
            for marker_id in marker_ids:
                self.marker_to_robot[marker_id] = robot_id
        
        # Detector ArUco
        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        
        # Captura de video
        self.input_video = None
        
        # Cargar calibración
        self.calibration_path = calibration_path
        self.cam_matrix, self.dist_coeffs = self.load_calibration_data()
        
        # Puntos 3D del marcador
        self.obj_points = np.array([
            [-marker_length/2,  marker_length/2, 0],
            [ marker_length/2,  marker_length/2, 0],
            [ marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)

    def load_calibration_data(self):
        """Carga los datos de calibración de la cámara"""
        data = np.load(self.calibration_path)
        cam_matrix = data["K"].astype(np.float32)
        dist_coeffs = data["D"].astype(np.float32)
        print("Matriz de calibración cargada:")
        print(cam_matrix)
        print("Coeficientes de distorsión:")
        print(dist_coeffs.ravel())
        return cam_matrix, dist_coeffs

    def detect_markers(self, image):
        """
        Detecta marcadores ArUco en una imagen.
        
        Args:
            image: Imagen donde detectar marcadores
            
        Returns:
            list[Marker]: Lista de marcadores detectados
        """
        corners, ids, rejected = self.detector.detectMarkers(image)
        markers = []
        base_marker = None

        if ids is not None:
            for i, c in enumerate(corners):
                marker_id = int(ids[i][0])
                center_x = int(np.mean(c[0][:, 0]))
                center_y = int(np.mean(c[0][:, 1]))

                # Calcular pose del marcador
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, c, self.cam_matrix, self.dist_coeffs
                )

                if success:
                    m = Marker(
                        id=marker_id,
                        corners=c,
                        center=(center_x, center_y),
                        rvec=rvec,
                        tvec=tvec
                    )
                    markers.append(m)

                    # Detectar marcador base (IDs 0 o 1). Preferimos 0 sobre 1 si ambos aparecen.
                    if marker_id in (0, 1):
                        if base_marker is None or marker_id == 0:
                            base_marker = m

        return markers, base_marker

    def get_robot_orientation(self, markers_dict, robot_id):
        """
        Calcula la orientación de un robot basándose en sus dos marcadores.
        
        Args:
            markers_dict: Diccionario {marker_id: Marker}
            robot_id: ID del robot
            
        Returns:
            float: Ángulo en grados (yaw) o None
        """
        marker_ids = self.robot_markers.get(robot_id, [])

        # Aceptar casos donde solo haya un marcador visible.
        present = [mid for mid in marker_ids if mid in markers_dict]
        if not present:
            return None

        # Usar el primer marcador disponible para obtener orientación
        marker = markers_dict[present[0]]
        R, _ = cv2.Rodrigues(marker.rvec)
        roll, pitch, yaw = self.rotation_matrix_to_euler_angles(R)
        return yaw

    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        """Convierte matriz de rotación a ángulos de Euler (roll, pitch, yaw)"""
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
        return np.degrees([x, y, z])

    @staticmethod
    def get_rt_matrix(R_, T_, force_type=-1):
        """Obtiene la matriz de transformación RT 4x4"""
        M = None
        R = R_.copy()
        T = T_.copy()
        
        if R.dtype == np.float64:
            Matrix = np.eye(4, dtype=np.float64)
            R33 = Matrix[0:3, 0:3]
            
            if R.size == 3:
                R33[:,:] = cv2.Rodrigues(R)[0]
            elif R.size == 9:
                R33[:,:] = R.astype(np.float64).reshape(3, 3)
            
            for i in range(3):
                Matrix[i, 3] = T.flat[i] if T.ndim > 1 else T[i]
            M = Matrix
            
        elif R.dtype == np.float32:
            Matrix = np.eye(4, dtype=np.float32)
            R33 = Matrix[0:3, 0:3]
            
            if R.size == 3:
                R33[:,:] = cv2.Rodrigues(R)[0]
            elif R.size == 9:
                R33[:,:] = R.astype(np.float32).reshape(3, 3)
            
            for i in range(3):
                Matrix[i, 3] = T.flat[i] if T.ndim > 1 else T[i]
            M = Matrix

        return M
        
    @staticmethod
    def calculate_distance_between_positions(tvec1, tvec2):
        """Calcula distancia 3D entre dos posiciones en metros"""
        pos1 = tvec1.flatten()
        pos2 = tvec2.flatten()
        return np.linalg.norm(pos1 - pos2)

    def calculate_angle_between_robots(self, markers_dict, robot_id1, robot_id2, base):
        """
        Calcula el ángulo relativo entre dos robots basándose en sus marcadores y un marcador base.

        Args:
            markers_dict: Diccionario {marker_id: Marker}
            robot_id1: ID del primer robot
            robot_id2: ID del segundo robot
            base: Marcador de referencia para el cálculo del ángulo

        Returns:
            dict: {'distance': float, 'angle1': float, 'angle2': float} o None
        """
        marker_ids1 = self.robot_markers.get(robot_id1, [])
        marker_ids2 = self.robot_markers.get(robot_id2, [])
        
        present1 = [mid for mid in marker_ids1 if mid in markers_dict]
        present2 = [mid for mid in marker_ids2 if mid in markers_dict]
        
        if not present1 or not present2:
            return None
        
        marker1 = markers_dict[present1[0]]
        marker2 = markers_dict[present2[0]]
        
        tvec_base = base.tvec.flatten()
        rvec_base = base.rvec
        tvec1 = marker1.tvec.flatten()
        tvec2 = marker2.tvec.flatten()
        rvec1 = marker1.rvec
        rvec2 = marker2.rvec
        
        # Construir matrices RT
        M1 = self.get_rt_matrix(rvec1, tvec1)
        M2 = self.get_rt_matrix(rvec2, tvec2)

        if M1 is None or M2 is None:
            raise ValueError("get_rt_matrix devolvió None para uno de los marcadores (revisar rvec/tvec y sus dtypes)")

        # Si no se proporciona un marcador base, trabajamos en el sistema de cámara/world tal como
        # lo devuelven las matrices RT de los marcadores (no hacemos transformación adicional).
        if base is None:
            M1_in_base = M1
            M2_in_base = M2
        else:
            M_Base = self.get_rt_matrix(rvec_base, tvec_base)
            if M_Base is None:
                raise ValueError("get_rt_matrix devolvió None para el marcador base (revisar rvec/tvec y sus dtypes)")

            # Transformar a sistema de coordenadas del marcador base
            MBase_inv = np.linalg.inv(M_Base)

            M1_in_base = MBase_inv @ M1
            M2_in_base = MBase_inv @ M2

        # Extraer vector de traslación (3x1) en el sistema base
        t1_in_base = M1_in_base[0:3, 3].reshape(3,1)
        t2_in_base = M2_in_base[0:3, 3].reshape(3,1)

        # Posiciciones en el sistema base
        pos_1 = M1_in_base[0:3, 3]
        pos_2 = M2_in_base[0:3, 3]
        v_12 = pos_2 - pos_1

        n_1 = M1_in_base[0:3, 2]  # eje Z del robot 1 en base
        n_2 = M2_in_base[0:3, 2]  # eje Z del robot 2 en base

        # Proyecciones de las normales y del vector entre robots en el plano XY del sistema base
        n_1_xy = np.array([n_1[0], n_1[1], 0.0])
        n_2_xy = np.array([n_2[0], n_2[1], 0.0])
        v_12_xy = np.array([v_12[0], v_12[1], 0.0])
        v_21_xy = -v_12_xy

        # Normalizar vectores
        def safe_normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-6 else v * 0.0
        
        n_1_xy = safe_normalize(n_1_xy)
        n_2_xy = safe_normalize(n_2_xy)
        v_12_xy = safe_normalize(v_12_xy)
        v_21_xy = safe_normalize(v_21_xy)

        # Ángulos (azimut) de cada vector
        az_n1 = math.atan2(n_1_xy[1], n_1_xy[0])
        az_v12 = math.atan2(v_12_xy[1], v_12_xy[0])
        az_n2 = math.atan2(n_2_xy[1], n_2_xy[0])
        az_v21 = math.atan2(v_21_xy[1], v_21_xy[0])

        # Diferencia angular en grados
        angle1 = self.normalize_angle_deg(np.degrees(az_v12 - az_n1))
        angle2 = self.normalize_angle_deg(np.degrees(az_v21 - az_n2))

        # Ajustar ángulos si es el aruco trasero
        if marker1.id == marker_ids1[0]:
            angle1 = self.normalize_angle_deg(angle1 + 180.0)
        if marker2.id == marker_ids2[0]:
            angle2 = self.normalize_angle_deg(angle2 + 180.0)

        # Distancia entre robots
        distance = self.calculate_distance_between_positions(t1_in_base, t2_in_base)
        
        return {
            'distance': distance,
            'angle1': angle1,
            'angle2': angle2
        }


    @staticmethod
    def normalize_angle_deg(a):
        """Normaliza ángulo a rango [-180, 180] grados"""
        return ((a + 180) % 360) - 180

    def process_frame(self, frame, base_marker=None):
        """
        Procesa un frame completo: detecta robots y calcula distancias/ángulos.
        
        Args:
            frame: Imagen capturada
            
        Returns:
            tuple: (frame_procesado, markers, robot_data, info_dict)
        """
        image_copy = frame.copy()
        
        # Detectar todos los marcadores
        markers, detected_base = self.detect_markers(frame)

        # Si no se pasó un base externo, usar el detectado en la imagen
        if base_marker is None:
            base_marker = detected_base
        
        # Crear diccionario de marcadores por ID
        markers_dict = {m.id: m for m in markers}
        
        # Información de robots detectados
        robot_data = {}  # {robot_id: {'center': (x,y), 'tvec': tvec, 'yaw': angle}}
        
        info = {
            'markers_detected': [m.id for m in markers],
            'robots_detected': [],
            'robot_pairs': []
        }

        # Dibujar marcadores detectados
        if len(markers) > 0:
            corners_list = [m.corners for m in markers]
            ids_array = np.array([[m.id] for m in markers])
            cv2.aruco.drawDetectedMarkers(image_copy, corners_list, ids_array)

            # Dibujar ejes de cada marcador
            for marker in markers:
                cv2.drawFrameAxes(image_copy, self.cam_matrix, self.dist_coeffs, 
                                 marker.rvec, marker.tvec, self.marker_length * 1.5)
        
        # Procesar cada robot
        for robot_id in self.robot_markers.keys():
            # IDs de marcadores de este robot
            marker_ids = self.robot_markers.get(robot_id, [])
            # Lista de Marker presentes (puede ser vacía, 1 o 2)
            present_markers = [markers_dict[mid] for mid in marker_ids if mid in markers_dict]

            if not present_markers:
                continue

            # Orientación (yaw) si necesitas mostrarla
            robot_yaw = self.get_robot_orientation(markers_dict, robot_id)

            # Centro del robot
            m = present_markers[0]
            cx, cy = int(m.center[0]), int(m.center[1])
            # Usar la tvec del marcador directamente como estimación del centro 3D del robot
            tvec = m.tvec

            # Guardar datos del robot
            robot_data[robot_id] = {
                'center': (cx, cy),
                'tvec': tvec,
                'yaw': robot_yaw if robot_yaw is not None else 0.0
            }
            info['robots_detected'].append(robot_id)

            # Dibujar centro del robot
            cv2.circle(image_copy, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(image_copy, f"Robot {robot_id}", 
                       (cx + 15, cy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if robot_yaw is not None:
                cv2.putText(image_copy, f"Yaw: {robot_yaw:.1f}°", 
                           (cx + 15, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Calcular distancias y ángulos entre robots
        detected_robots = list(robot_data.keys())
        
        if len(detected_robots) >= 2:
            # Calcular para todos los pares posibles
            for i in range(len(detected_robots)):
                for j in range(i + 1, len(detected_robots)):
                    robot_id1 = detected_robots[i]
                    robot_id2 = detected_robots[j]
                    
                    # Calcular ángulo relativo y distancia usando markers_dict
                    angle_info = self.calculate_angle_between_robots(markers_dict, robot_id1, robot_id2, base_marker)
                    if angle_info is None:
                        continue

                    pair_info = {
                        'robot1': robot_id1,
                        'robot2': robot_id2,
                        'distance': angle_info['distance'],
                        'angle1': angle_info['angle1'],
                        'angle2': angle_info['angle2'],
                    }
                    
                    info['robot_pairs'].append(pair_info)
                    
                    # Dibujar línea entre robots
                    pt1 = robot_data[robot_id1]['center']
                    pt2 = robot_data[robot_id2]['center']
                    cv2.line(image_copy, pt1, pt2, (255, 255, 0), 2)
                    
                    # Mostrar información en el punto medio
                    mid_x = (pt1[0] + pt2[0]) // 2
                    mid_y = (pt1[1] + pt2[1]) // 2
                    
                    cv2.putText(image_copy, 
                               f"D:{pair_info['distance']*100:.1f}cm A:{pair_info['angle1']:.1f}°",
                               (mid_x, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Información en pantalla
        y_offset = 30
        cv2.putText(image_copy, f"Marcadores: {len(markers)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(image_copy, f"Robots: {len(detected_robots)}/{len(self.robot_markers)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image_copy, markers, robot_data, info

    def run(self, video_source=None):
        """
        Ejecuta el detector en modo interactivo.
        
        Args:
            video_source: Ruta de video o None para usar cámara
        """
        if video_source:
            self.input_video = cv2.VideoCapture(video_source)
        else:
            self.input_video = cv2.VideoCapture(self.cam_id)
        
        print("Controles:")
        print("  ESC - Salir")
        
        while True:
            ret, frame = self.input_video.read()
            if not ret:
                break
            
            frame_processed, markers, robot_data, info = self.process_frame(frame)
            
            cv2.imshow("Detector ArUco - Robots", frame_processed)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        self.release()

    def release(self):
        """Libera recursos"""
        if self.input_video:
            self.input_video.release()
        cv2.destroyAllWindows()