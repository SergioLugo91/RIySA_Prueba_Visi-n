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
    def __init__(self, marker_length=0.025, cam_id=0, target_fps=30.0,
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
                    markers.append(Marker(
                        id=marker_id,
                        corners=c,
                        center=(center_x, center_y),
                        rvec=rvec,
                        tvec=tvec
                    ))
        
        return markers

    def get_robot_center(self, markers_dict, robot_id):
        """
        Calcula el centro de un robot usando sus dos marcadores.
        
        Args:
            markers_dict: Diccionario {marker_id: Marker}
            robot_id: ID del robot
            
        Returns:
            tuple: (center_x, center_y, tvec_avg) o None si no se detectan ambos marcadores
        """
        marker_ids = self.robot_markers.get(robot_id, [])
        
        if len(marker_ids) != 2:
            return None
        
        # Verificar que ambos marcadores estén detectados
        if marker_ids[0] not in markers_dict or marker_ids[1] not in markers_dict:
            return None
        
        marker1 = markers_dict[marker_ids[0]]
        marker2 = markers_dict[marker_ids[1]]
        
        # Calcular centro promedio en imagen
        center_x = (marker1.center[0] + marker2.center[0]) / 2
        center_y = (marker1.center[1] + marker2.center[1]) / 2
        
        # Calcular posición 3D promedio
        tvec_avg = (marker1.tvec + marker2.tvec) / 2
        
        return (int(center_x), int(center_y), tvec_avg)

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
        
        if len(marker_ids) != 2:
            return None
        
        if marker_ids[0] not in markers_dict or marker_ids[1] not in markers_dict:
            return None
        
        marker1 = markers_dict[marker_ids[0]]
        marker2 = markers_dict[marker_ids[1]]
        
        # Usar el primer marcador para obtener orientación
        R, _ = cv2.Rodrigues(marker1.rvec)
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
        
        return M if force_type == -1 else M.astype(force_type)

    @staticmethod
    def calculate_distance_between_positions(tvec1, tvec2):
        """Calcula distancia 3D entre dos posiciones en metros"""
        pos1 = tvec1.flatten()
        pos2 = tvec2.flatten()
        return np.linalg.norm(pos1 - pos2)

    @staticmethod
    def calculate_angle_between_robots(tvec1, tvec2):
        """
        Calcula el ángulo relativo entre dos robots.
        
        Args:
            tvec1, tvec2: Vectores de posición 3D
            
        Returns:
            float: Ángulo en grados
        """
        # Vector del robot 1 al robot 2
        delta = tvec2.flatten() - tvec1.flatten()
        
        # Ángulo en el plano XY (asumiendo que Z es arriba)
        angle_rad = math.atan2(delta[1], delta[0])
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg

    @staticmethod
    def normalize_angle_deg(a):
        """Normaliza ángulo a rango [-180, 180] grados"""
        return ((a + 180) % 360) - 180

    def process_frame(self, frame):
        """
        Procesa un frame completo: detecta robots y calcula distancias/ángulos.
        
        Args:
            frame: Imagen capturada
            
        Returns:
            tuple: (frame_procesado, markers, robot_data, info_dict)
        """
        image_copy = frame.copy()
        
        # Detectar todos los marcadores
        markers = self.detect_markers(frame)
        
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
            robot_center = self.get_robot_center(markers_dict, robot_id)
            robot_yaw = self.get_robot_orientation(markers_dict, robot_id)
            
            if robot_center is not None:
                cx, cy, tvec = robot_center
                
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
                    
                    tvec1 = robot_data[robot_id1]['tvec']
                    tvec2 = robot_data[robot_id2]['tvec']
                    
                    # Calcular distancia
                    distance = self.calculate_distance_between_positions(tvec1, tvec2)
                    
                    # Calcular ángulo relativo
                    angle = self.calculate_angle_between_robots(tvec1, tvec2)
                    
                    pair_info = {
                        'robot1': robot_id1,
                        'robot2': robot_id2,
                        'distance': distance,
                        'angle': angle
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
                               f"D:{distance*100:.1f}cm A:{angle:.1f}°",
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