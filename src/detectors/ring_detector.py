import cv2                  
import numpy as np                  
import cv2.aruco as aruco  
from collections import deque
from filters.median_corner_filter import MedianCornerFilter


class RingDetector:
    """
    Detector del ring de combate (cuadrilátero) usando marcadores ArUco.
    Calcula la homografía entre la imagen de la cámara y las coordenadas reales del ring.
    """

    def __init__(self, width=0.80, height=0.80, marker_length=0.09, 
                 id_a=0, id_b=1, 
                 offset_a=(0.0, 0.03), offset_b=(0.0, 0.05),
                 cam_id=1, target_fps=30.0,
                 calibration_path="calibracion/cam_calib_data.npz"):
        """
        Constructor de la clase RingDetector.
        Inicializa parámetros geométricos, cámara, detector ArUco y filtros.
        """

        # Dimensiones reales del ring (en metros)
        self.width = width
        self.height = height

        # Tamaño del lado del marcador ArUco
        self.marker_length = marker_length
        
        # IDs de los dos marcadores de referencia
        self.id_a = id_a
        self.id_b = id_b

        # Offsets de los marcadores respecto a las esquinas del ring
        self.ox_a, self.oy_a = offset_a
        self.ox_b, self.oy_b = offset_b
        
        # ID de la cámara a utilizar
        self.cam_id = cam_id

        # FPS objetivo
        self.target_fps = target_fps

        # Periodo entre frames (no se usa explícitamente, pero queda definido)
        self.frame_period = 1.0 / target_fps
        
        # Filtro de mediana para suavizar las esquinas detectadas del ring
        self.corner_filter = MedianCornerFilter(num_samples=10)
        
        # Diccionario ArUco (6x6, 250 IDs posibles)
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_6X6_250
        )

        # Parámetros del detector ArUco
        self.detector_params = cv2.aruco.DetectorParameters()

        # Detector ArUco propiamente dicho
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary, self.detector_params
        )
        
        # Inicializa la captura de vídeo desde la cámara
        self.input_video = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        
        # Ruta al archivo de calibración
        self.calibration_path = calibration_path

        # Carga matriz intrínseca y coeficientes de distorsión
        self.cam_matrix, self.dist_coeffs = self.load_calibration_data()
        
        # Variables de estado
        self.current_homography = None   # Última homografía válida
        self.ring_corners = None         # Últimas esquinas del ring detectadas


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
        Devuelve las coordenadas reales de las 4 esquinas del marcador
        situado cerca del origen (0,0) del ring.
        """

        return np.array([
            [-ox - s, -oy - s],  # Esquina superior izquierda
            [ox, -oy - s],       # Esquina superior derecha
            [ox, -oy],           # Esquina inferior derecha
            [ox - s, -oy]        # Esquina inferior izquierda
        ], dtype=np.float32)


    def world_corners_for_marker_near_WH(self, ox, oy, s):
        """
        Devuelve las coordenadas reales de las 4 esquinas del marcador
        situado cerca de la esquina (W,H) del ring.
        """

        return np.array([
            [self.width + ox, self.height + oy],          # TL
            [self.width + ox + s, self.height + oy],      # TR
            [self.width + ox + s, self.height + oy + s],  # BR
            [self.width + ox, self.height + oy + s]       # BL
        ], dtype=np.float32)


    def img_to_world(self, Hmat, u, v):
        """
        Convierte un punto de la imagen (u,v) a coordenadas del mundo
        usando la homografía.
        """

        # Punto en coordenadas homogéneas
        p = np.array([u, v, 1.0], dtype=np.float32)

        # Aplica la homografía
        q = Hmat @ p

        # Normaliza coordenadas homogéneas
        q /= q[2]

        return float(q[0]), float(q[1])


    def process_frame(self, frame):
        """
        Procesa un frame de la cámara:
        - Corrige distorsión
        - Detecta marcadores ArUco
        - Calcula homografía
        - Proyecta el ring
        - Estima posición en coordenadas reales
        """

        # Corrige la distorsión de la lente
        frame_und = cv2.undistort(frame, self.cam_matrix, self.dist_coeffs)

        # Convierte a escala de grises
        gray = cv2.cvtColor(frame_und, cv2.COLOR_BGR2GRAY)
        
        # Detecta marcadores ArUco
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        # Inicializa variables de salida
        both_markers_detected = False
        Hmat = None

        # Diccionario con información del estado
        info = {
            'markers_detected': [],
            'both_markers': False,
            'position_valid': False,
            'world_coords': None,
            'inside_ring': False
        }
        
        # Si se ha detectado al menos un marcador
        if ids is not None:

            # Dibuja los marcadores detectados
            aruco.drawDetectedMarkers(frame_und, corners, ids)
            
            # Para cada marcador detectado
            for i, mid in enumerate(ids.flatten()):

                # Coordenadas de las esquinas del marcador
                pts = corners[i][0]

                # Calcula el centro del marcador
                center = pts.mean(axis=0)
                cx, cy = int(center[0]), int(center[1])

                # Dibuja el centro
                cv2.circle(frame_und, (cx, cy), 4, (0, 255, 0), -1)

                # Escribe el ID del marcador
                cv2.putText(frame_und, f"ID {int(mid)}", (cx + 8, cy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Guarda el ID detectado
                info['markers_detected'].append(int(mid))
            
            # Comprueba si están presentes ambos marcadores
            ids_flat = ids.flatten()
            has_marker_a = self.id_a in ids_flat
            has_marker_b = self.id_b in ids_flat

            both_markers_detected = has_marker_a and has_marker_b
            info['both_markers'] = both_markers_detected
            
            if both_markers_detected:
                # Listas de correspondencias imagen ↔ mundo
                img_pts = []
                world_pts = []
                
                # Asocia esquinas de imagen con coordenadas reales
                for i, mid in enumerate(ids_flat):
                    if mid == self.id_a:
                        img_pts.extend(corners[i][0])
                        world_pts.extend(
                            self.world_corners_for_marker_near_00(
                                self.ox_a, self.oy_a, self.marker_length
                            )
                        )
                    elif mid == self.id_b:
                        img_pts.extend(corners[i][0])
                        world_pts.extend(
                            self.world_corners_for_marker_near_WH(
                                self.ox_b, self.oy_b, self.marker_length
                            )
                        )
                
                # Si hay suficientes puntos, calcula homografía
                if len(img_pts) >= 8:
                    img_pts = np.array(img_pts, dtype=np.float32)
                    world_pts = np.array(world_pts, dtype=np.float32)

                    Hmat, _ = cv2.findHomography(
                        img_pts, world_pts,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0
                    )
                    
                    if Hmat is not None:
                        # Guarda homografía actual
                        self.current_homography = Hmat

                        # Inversa para proyectar ring a imagen
                        H_inv = np.linalg.inv(Hmat)
                        
                        # Esquinas reales del ring
                        ring_world = np.array([
                            [0, 0, 1],
                            [self.width, 0, 1],
                            [self.width, self.height, 1],
                            [0, self.height, 1]
                        ], dtype=np.float32).T
                        
                        # Proyección a imagen
                        ring_img = H_inv @ ring_world
                        ring_img /= ring_img[2]

                        ring_corners_raw = ring_img[:2].T.astype(np.float32)
                        
                        # Añade muestra al filtro de mediana
                        self.corner_filter.add_sample(ring_corners_raw)
                        
                        # Calcula la posición del centro de la imagen en el mundo
                        h_, w_ = frame_und.shape[:2]
                        xw, yw = self.img_to_world(Hmat, w_/2, h_/2)
                        
                        # Comprueba si está dentro del ring
                        inside = (0 <= xw <= self.width) and (0 <= yw <= self.height)
                        
                        info['position_valid'] = True
                        info['world_coords'] = (xw, yw)
                        info['inside_ring'] = inside
                        
                        # Muestra información en pantalla
                        cv2.putText(
                            frame_und,
                            f"x={xw:.3f}m  y={yw:.3f}m  {'DENTRO' if inside else 'FUERA'}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if inside else (0, 0, 255), 2
                        )
            else:
                # Indica qué marcador falta
                missing = []
                if not has_marker_a:
                    missing.append(f"ID {self.id_a}")
                if not has_marker_b:
                    missing.append(f"ID {self.id_b}")
                
                cv2.putText(
                    frame_und,
                    f"Falta marcador: {', '.join(missing)}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 165, 255), 2
                )
        
        # Obtiene esquinas suavizadas por mediana
        ring_corners_median = self.corner_filter.get_median_corners()
        
        if ring_corners_median is not None:
            self.ring_corners = ring_corners_median
            poly = ring_corners_median.reshape(-1, 1, 2).astype(np.int32)
            
            # Dibuja el ring según el estado del filtro
            if self.corner_filter.is_using_cached_value():
                cv2.polylines(frame_und, [poly], True, (0, 0, 255), 2)
            elif self.corner_filter.is_ready():
                cv2.polylines(frame_und, [poly], True, (0, 255, 0), 2)
            else:
                cv2.polylines(frame_und, [poly], True, (0, 255, 255), 2)
        
        return frame_und, Hmat, ring_corners_median, info


    def run(self):
        """Bucle principal del sistema"""

        while True:
            ok, frame = self.input_video.read()
            if not ok:
                break
            
            frame_processed, homography, corners, info = self.process_frame(frame)
            cv2.imshow("Detección Ring de Combate", frame_processed)
            
            # ESC para salir
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.release()


    def get_current_homography(self):
        """Devuelve la homografía actual"""
        return self.current_homography


    def get_ring_corners(self):
        """Devuelve las esquinas actuales del ring en la imagen"""
        return self.ring_corners


    def release(self):
        """Libera la cámara y cierra ventanas"""
        self.input_video.release()
        cv2.destroyAllWindows()
