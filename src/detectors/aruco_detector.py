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
    def __init__(self, marker_length=0.076, cam_id=0, target_fps=30.0,
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
            0: [2, 3],    # Robot 0 usa ArUco IDs 2 y 3
            1: [6, 7],    # Robot 1 usa ArUco IDs 4 y 5
            2: [4, 5],    # Robot 2 usa ArUco IDs 6 y 7
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
        cam_matrix = np.array([
            [811.190329608064, 0, 304.044574492494],
            [0, 807.950042818991, 224.991673688224],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.array([0.000464623904805219, -0.0394572576121102, 0, 0, 0], dtype=np.float32)
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        Calcula la orientación de un robot basándose en alguno de sus marcadores.
        
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
        
        # Busca los marcadores presentes de cada robot
        present1 = [mid for mid in marker_ids1 if mid in markers_dict]
        present2 = [mid for mid in marker_ids2 if mid in markers_dict]
        
        # Si no hay marcadores presentes para alguno de los robots, no se puede calcular
        if not present1 or not present2:
            return None
        
        marker1 = markers_dict[present1[0]]
        marker2 = markers_dict[present2[0]]
        
        try:
            if base is not None:
                tvec_base = base.tvec
                rvec_base = base.rvec
            tvec1 = marker1.tvec
            tvec2 = marker2.tvec
            rvec1 = marker1.rvec
            rvec2 = marker2.rvec
        except Exception as e:
            print(f"[ERROR] Al obtener tvec/rvec de marcadores: {e}")
            return None
        
        # Construir matrices RT
        M1 = self.get_rt_matrix(rvec1, tvec1)
        M2 = self.get_rt_matrix(rvec2, tvec2)

        if M1 is None or M2 is None:
            raise ValueError("get_rt_matrix devolvió None para uno de los marcadores (revisar rvec/tvec y sus dtypes)")

        # Si no se proporciona un marcador base externo, no se puede calcular el ángulo
        if base is None:
            print(f"[ERROR] No hay base para calcular ángulos entre robots.")
            return None
        else:
            M_Base = self.get_rt_matrix(rvec_base, tvec_base)
            if M_Base is None:
                raise ValueError("get_rt_matrix devolvió None para el marcador base (revisar rvec/tvec y sus dtypes)")

            # Transformar a sistema de coordenadas del marcador base
            MBase_inv = np.linalg.inv(M_Base)

            M1_in_base = MBase_inv @ M1
            M2_in_base = MBase_inv @ M2
            #print("M1 en coordenadas base:")
            #print(M1_in_base)
            #print("M2 en coordenadas base: ")
            #print(M2_in_base)

        # Extraer vector de traslación (3x1) en el sistema base
        t1_in_base = M1_in_base[0:3, 3].reshape(3,1)
        t2_in_base = M2_in_base[0:3, 3].reshape(3,1)

        # Posiciciones en el sistema base
        pos_1 = M1_in_base[0:3, 3]
        pos_2 = M2_in_base[0:3, 3]
        vector_12 = pos_2 - pos_1

        n_1 = M1_in_base[0:3, 2]  # eje Z del robot 1 en base
        n_2 = M2_in_base[0:3, 2]  # eje Z del robot 2 en base

        # Proyecciones de las normales y del vector entre robots en el plano XY del sistema base
        n_1_xy = np.array([n_1[0], n_1[1], 0.0])
        n_2_xy = np.array([n_2[0], n_2[1], 0.0])
        v_12_xy = np.array([vector_12[0], vector_12[1], 0.0])
        v_21_xy = -v_12_xy

        # Normalizar vectores
        def safe_normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-6 else v * 0.0
        
        n_1_xy = safe_normalize(n_1_xy)
        n_2_xy = safe_normalize(n_2_xy)
        v_12_xy = safe_normalize(v_12_xy)
        v_21_xy = safe_normalize(v_21_xy)

        # Preparar vectores "frente" por robot (si el marcador es trasero, se invierte)
        is_rear1 = (marker1.id == marker_ids1[0])
        is_rear2 = (marker2.id == marker_ids2[0])

        if is_rear1:
            print(f"[DEBUG] Marker trasero Robot {robot_id1} detectado (ID {marker1.id})")
        if is_rear2:
            print(f"[DEBUG] Marker trasero Robot {robot_id2} detectado (ID {marker2.id})")

        f1_xy = -n_1_xy if is_rear1 else n_1_xy
        f2_xy = -n_2_xy if is_rear2 else n_2_xy

        # Ángulo firmado entre el frente del robot y el vector hacia el otro robot
        def signed_angle_deg(a_xy, b_xy):
            cross_z = a_xy[0]*b_xy[1] - a_xy[1]*b_xy[0]
            dot_ab  = a_xy[0]*b_xy[0] + a_xy[1]*b_xy[1]
            return np.degrees(math.atan2(cross_z, dot_ab))

        # Robot 1: frente f1 con vector hacia 2
        angle_r1 = signed_angle_deg(f1_xy, v_12_xy)
        # Robot 2: frente f2 con vector hacia 1
        angle_r2 = signed_angle_deg(f2_xy, v_21_xy)

        # Ajuste de convención de signo: en OpenCV el eje Y crece hacia abajo,
        # lo que invierte el signo si se desea convención matemática (Y hacia arriba).
        angle_r1 = -angle_r1
        angle_r2 = -angle_r2

        # Normalizar a rango [-180, 180] para ambos ángulos
        angle_r1 = self.normalize_angle_deg(angle_r1)
        angle_r2 = self.normalize_angle_deg(angle_r2)

        # Distancia entre robots
        distance = self.calculate_distance_between_positions(t1_in_base, t2_in_base)
        distance_cm = distance * 100  # Convertir de metros a centímetros
        #print(f"[DEBUG] Distancia entre Robot {robot_id1} y Robot {robot_id2}: {distance_cm:.2f} cm, Ángulo1: {angle1:.2f}°, Ángulo2: {angle2:.2f}°")

        return {
            'distance': distance_cm,
            'angle1': angle_r1,
            'angle2': angle_r2
        }


    @staticmethod
    def normalize_angle_deg(a):
        """Normaliza ángulo a rango [-180, 180] grados"""
        return ((a + 180) % 360) - 180

    @staticmethod
    def select_active_pair(robot_pairs, ubots):
        """Selecciona el par a usar según robots dentro/fuera.

        Regla:
        - Si hay al menos dos con Out == 0, usa el primer par entre dos 'dentro'.
        - Si solo hay uno dentro, usa el primer par que lo conecta con otro robot.
        - Si no hay dentro, no selecciona par.
        """
        inside = {rid for rid, u in ubots.items() if getattr(u, 'Out', 1) == 0}
        if len(inside) >= 2:
            for p in robot_pairs:
                if p['robot1'] in inside and p['robot2'] in inside:
                    return p
        elif len(inside) == 1:
            inside_id = next(iter(inside))
            for p in robot_pairs:
                if p['robot1'] == inside_id or p['robot2'] == inside_id:
                    return p
        return None

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
            #for marker in markers:
            #    cv2.drawFrameAxes(image_copy, self.cam_matrix, self.dist_coeffs, 
            #                     marker.rvec, marker.tvec, self.marker_length * 1.5)
        
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