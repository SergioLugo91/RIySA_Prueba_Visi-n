import cv2                  
import numpy as np                  
import cv2.aruco as aruco  
from collections import deque
from filters.median_corner_filter import MedianCornerFilter

class RingDetector:
    """
    Detector del ring de combate (cuadrilátero) usando marcadores ArUco.
    Calcula la homografía entre la imagen y las coordenadas del mundo (ring).
    """
    def __init__(self, width=0.80, height=0.80, marker_length=0.09, 
                 id_a=0, id_b=1, 
                 offset_a=(0.0, 0.03), offset_b=(0.0, 0.05),
                 cam_id=1, target_fps=30.0,
                 calibration_path="calibracion/cam_calib_data.npz"):
        """
        Args:
            width (float): Ancho del ring en metros
            height (float): Alto del ring en metros
            marker_length (float): Longitud del lado del marcador ArUco en metros
            id_a (int): ID del marcador cerca de la esquina (0,0)
            id_b (int): ID del marcador cerca de la esquina (W,H)
            offset_a (tuple): Offsets (ox, oy) del marcador A hacia el exterior
            offset_b (tuple): Offsets (ox, oy) del marcador B hacia el exterior
            cam_id (int): ID de la cámara
            target_fps (float): FPS objetivo
            calibration_path (str): Ruta al archivo de calibración
        """
        # Dimensiones del ring
        self.width = width
        self.height = height
        self.marker_length = marker_length
        
        # IDs y offsets de los marcadores
        self.id_a = id_a
        self.id_b = id_b
        self.ox_a, self.oy_a = offset_a
        self.ox_b, self.oy_b = offset_b
        
        # Configuración de cámara
        self.cam_id = cam_id
        self.target_fps = target_fps
        self.frame_period = 1.0 / target_fps
        
        # Filtro de mediana para suavizar las esquinas del ring
        self.corner_filter = MedianCornerFilter(num_samples=10)
        
        # Detector ArUco
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        
        # Captura de video
        self.input_video = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        
        # Cargar calibración
        self.calibration_path = calibration_path
        self.cam_matrix, self.dist_coeffs = self.load_calibration_data()
        
        # Estado actual
        self.current_homography = None
        self.ring_corners = None

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

    def world_corners_for_marker_near_00(self, ox, oy, s):
        """
        Devuelve las 4 esquinas del marcador cerca de (0,0) en coordenadas del mundo.
        
        Args:
            ox, oy: Offsets hacia el exterior desde la esquina
            s: Lado del marcador en metros
            
        Returns:
            np.ndarray: Array (4, 2) con las esquinas [TL, TR, BR, BL]
        """
        return np.array([
            [-ox - s, -oy - s],  # TL (Top-Left)
            [ox, -oy - s],       # TR (Top-Right)
            [ox, -oy],           # BR (Bottom-Right)
            [ox - s, -oy]        # BL (Bottom-Left)
        ], dtype=np.float32)

    def world_corners_for_marker_near_WH(self, ox, oy, s):
        """
        Devuelve las 4 esquinas del marcador cerca de (W,H) en coordenadas del mundo.
        
        Args:
            ox, oy: Offsets hacia el exterior desde la esquina
            s: Lado del marcador en metros
            
        Returns:
            np.ndarray: Array (4, 2) con las esquinas [TL, TR, BR, BL]
        """
        return np.array([
            [self.width + ox, self.height + oy],          # TL
            [self.width + ox + s, self.height + oy],      # TR
            [self.width + ox + s, self.height + oy + s],  # BR
            [self.width + ox, self.height + oy + s]       # BL
        ], dtype=np.float32)

    def img_to_world(self, Hmat, u, v):
        """
        Proyecta un punto de imagen (u,v) a coordenadas del mundo usando homografía.
        
        Args:
            Hmat: Matriz de homografía 3x3
            u, v: Coordenadas en píxeles
            
        Returns:
            tuple: (x, y) en coordenadas del ring (metros)
        """
        p = np.array([u, v, 1.0], dtype=np.float32)
        q = Hmat @ p
        q /= q[2]
        return float(q[0]), float(q[1])

    def process_frame(self, frame):
        """
        Procesa un frame para detectar el ring y calcular la homografía.
        
        Args:
            frame: Imagen capturada de la cámara
            
        Returns:
            tuple: (frame_procesado, homografia, esquinas_ring, info_dict)
        """
        # Corregir distorsión
        frame_und = cv2.undistort(frame, self.cam_matrix, self.dist_coeffs)
        gray = cv2.cvtColor(frame_und, cv2.COLOR_BGR2GRAY)
        
        # Detectar marcadores
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        both_markers_detected = False
        Hmat = None
        info = {
            'markers_detected': [],
            'both_markers': False,
            'position_valid': False,
            'world_coords': None,
            'inside_ring': False
        }
        
        if ids is not None:
            # Dibujar marcadores detectados
            aruco.drawDetectedMarkers(frame_und, corners, ids)
            
            # Dibujar centro e ID de cada marcador
            for i, mid in enumerate(ids.flatten()):
                pts = corners[i][0]
                center = pts.mean(axis=0)
                cx, cy = int(center[0]), int(center[1])
                cv2.circle(frame_und, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame_und, f"ID {int(mid)}", (cx + 8, cy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                info['markers_detected'].append(int(mid))
            
            # Verificar si ambos marcadores están presentes
            ids_flat = ids.flatten()
            has_marker_a = self.id_a in ids_flat
            has_marker_b = self.id_b in ids_flat
            both_markers_detected = has_marker_a and has_marker_b
            info['both_markers'] = both_markers_detected
            
            if both_markers_detected:
                # Construir correspondencias imagen <-> mundo
                img_pts = []
                world_pts = []
                
                for i, mid in enumerate(ids_flat):
                    if mid == self.id_a:
                        img_pts.extend(corners[i][0])
                        world_pts.extend(self.world_corners_for_marker_near_00(
                            self.ox_a, self.oy_a, self.marker_length))
                    elif mid == self.id_b:
                        img_pts.extend(corners[i][0])
                        world_pts.extend(self.world_corners_for_marker_near_WH(
                            self.ox_b, self.oy_b, self.marker_length))
                
                # Calcular homografía
                if len(img_pts) >= 8:
                    img_pts = np.array(img_pts, dtype=np.float32)
                    world_pts = np.array(world_pts, dtype=np.float32)
                    Hmat, _ = cv2.findHomography(
                        img_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    
                    if Hmat is not None:
                        self.current_homography = Hmat
                        H_inv = np.linalg.inv(Hmat)
                        
                        # Proyectar esquinas del ring a imagen
                        ring_world = np.array([
                            [0, 0, 1],
                            [self.width, 0, 1],
                            [self.width, self.height, 1],
                            [0, self.height, 1]
                        ], dtype=np.float32).T
                        
                        ring_img = H_inv @ ring_world
                        ring_img /= ring_img[2]
                        ring_corners_raw = ring_img[:2].T.astype(np.float32)
                        
                        # Añadir al filtro de mediana
                        self.corner_filter.add_sample(ring_corners_raw)
                        
                        # Calcular posición del centro de imagen en el mundo
                        h_, w_ = frame_und.shape[:2]
                        xw, yw = self.img_to_world(Hmat, w_/2, h_/2)
                        
                        # Verificar si está dentro del ring
                        inside = (0 <= xw <= self.width) and (0 <= yw <= self.height)
                        
                        info['position_valid'] = True
                        info['world_coords'] = (xw, yw)
                        info['inside_ring'] = inside
                        
                        # Dibujar información de posición
                        cv2.putText(
                            frame_und,
                            f"x={xw:.3f}m  y={yw:.3f}m  {'DENTRO' if inside else 'FUERA'}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if inside else (0, 0, 255), 2
                        )
            else:
                # Mostrar marcadores faltantes
                missing = []
                if not has_marker_a:
                    missing.append(f"ID {self.id_a}")
                if not has_marker_b:
                    missing.append(f"ID {self.id_b}")
                
                cv2.putText(frame_und, f"Falta marcador: {', '.join(missing)}", 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Obtener esquinas suavizadas
        ring_corners_median = self.corner_filter.get_median_corners()
        
        if ring_corners_median is not None:
            self.ring_corners = ring_corners_median
            poly = ring_corners_median.reshape(-1, 1, 2).astype(np.int32)
            
            # Dibujar el cuadrilátero del ring con colores según estado
            if self.corner_filter.is_using_cached_value():
                cv2.polylines(frame_und, [poly], True, (0, 0, 255), 2)
                cv2.putText(frame_und, 
                           f"CACHE - Sin detección: {self.corner_filter.frames_without_detection} frames",
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif self.corner_filter.is_ready():
                cv2.polylines(frame_und, [poly], True, (0, 255, 0), 2)
            else:
                cv2.polylines(frame_und, [poly], True, (0, 255, 255), 2)
                cv2.putText(frame_und, 
                           f"Inicializando: {len(self.corner_filter.buffer)}/{self.corner_filter.num_samples}",
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame_und, Hmat, ring_corners_median, info

    def run(self):
        """Ejecuta el detector en modo interactivo (bucle principal)"""
        while True:
            ok, frame = self.input_video.read()
            if not ok:
                break
            
            frame_processed, homography, corners, info = self.process_frame(frame)
            
            cv2.imshow("Detección Ring de Combate", frame_processed)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                break
        
        self.release()

    def get_current_homography(self):
        """Retorna la homografía actual (o None si no está disponible)"""
        return self.current_homography

    def get_ring_corners(self):
        """Retorna las esquinas actuales del ring en la imagen"""
        return self.ring_corners

    def release(self):
        """Libera recursos"""
        self.input_video.release()
        cv2.destroyAllWindows()