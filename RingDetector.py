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
        self.num_samples = num_samples
        self.buffer = deque(maxlen=num_samples)
        self.last_known_corners = None
        self.frames_without_detection = 0
        self.max_frames_without_detection = 30
    
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


class RingDetector:
    """Detector del ring de combate usando ArUco"""
    
    def __init__(self, width=0.80, height=0.80, marker_len=0.09, 
                 id_a=0, id_b=1, ox_a=0.0, oy_a=0.03, ox_b=0.0, oy_b=0.05,
                 calibration_path="Calibracion/cam_calib_data.npz"):
        self.WIDTH = width
        self.HEIGHT = height
        self.marker_len = marker_len
        self.ID_A = id_a
        self.ID_B = id_b
        self.ox_A = ox_a
        self.oy_A = oy_a
        self.ox_B = ox_b
        self.oy_B = oy_b
        
        # Detector ArUco
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detectorParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detectorParams)
        
        # Filtro de mediana
        self.corner_filter = MedianCornerFilter(num_samples=10)
        
        # Cargar calibración
        data = np.load(calibration_path)
        self.camMatrix = data["K"].astype(np.float32)
        self.distCoeffs = data["D"].astype(np.float32)
        
        print("RingDetector: Calibración cargada")
    
    def world_corners_for_marker_near_00(self):
        s = self.marker_len
        return np.array([
            [-self.ox_A - s, -self.oy_A - s],
            [self.ox_A, -self.oy_A - s],
            [self.ox_A, -self.oy_A],
            [self.ox_A - s, -self.oy_A]
        ], dtype=np.float32)
    
    def world_corners_for_marker_near_WH(self):
        s = self.marker_len
        return np.array([
            [self.WIDTH + self.ox_B, self.HEIGHT + self.oy_B],
            [self.WIDTH + self.ox_B + s, self.HEIGHT + self.oy_B],
            [self.WIDTH + self.ox_B + s, self.HEIGHT + self.oy_B + s],
            [self.WIDTH + self.ox_B, self.HEIGHT + self.oy_B + s]
        ], dtype=np.float32)
    
    def img_to_world(self, Hmat, u, v):
        p = np.array([u, v, 1.0], dtype=np.float32)
        q = Hmat @ p
        q /= q[2]
        return float(q[0]), float(q[1])
    
    def process_frame(self, frame):
        """
        Procesa un frame y devuelve información del ring.
        
        Returns:
            dict: {
                'frame': frame procesado,
                'homography': matriz de homografía (o None),
                'corners': esquinas del ring suavizadas,
                'both_detected': bool,
                'inside': bool (si el centro está dentro),
                'world_pos': (x, y) en metros
            }
        """
        frame_und = cv2.undistort(frame, self.camMatrix, self.distCoeffs)
        gray = cv2.cvtColor(frame_und, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        result = {
            'frame': frame_und,
            'homography': None,
            'corners': None,
            'both_detected': False,
            'inside': False,
            'world_pos': None
        }
        
        if ids is not None:
            aruco.drawDetectedMarkers(frame_und, corners, ids)
            
            for i, mid in enumerate(ids.flatten()):
                pts = corners[i][0]
                center = pts.mean(axis=0)
                cx, cy = int(center[0]), int(center[1])
                cv2.circle(frame_und, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame_und, f"ID {int(mid)}", (cx + 8, cy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            ids_flat = ids.flatten()
            has_marker_A = self.ID_A in ids_flat
            has_marker_B = self.ID_B in ids_flat
            result['both_detected'] = has_marker_A and has_marker_B
            
            if result['both_detected']:
                img_pts = []
                world_pts = []
                
                for i, mid in enumerate(ids_flat):
                    if mid == self.ID_A:
                        img_pts.extend(corners[i][0])
                        world_pts.extend(self.world_corners_for_marker_near_00())
                    elif mid == self.ID_B:
                        img_pts.extend(corners[i][0])
                        world_pts.extend(self.world_corners_for_marker_near_WH())
                
                if len(img_pts) >= 8:
                    img_pts = np.array(img_pts, dtype=np.float32)
                    world_pts = np.array(world_pts, dtype=np.float32)
                    Hmat, _ = cv2.findHomography(img_pts, world_pts, 
                                                 method=cv2.RANSAC, ransacReprojThreshold=3.0)
                    
                    if Hmat is not None:
                        result['homography'] = Hmat
                        H_inv = np.linalg.inv(Hmat)
                        
                        ring_world = np.array([
                            [0, 0, 1],
                            [self.WIDTH, 0, 1],
                            [self.WIDTH, self.HEIGHT, 1],
                            [0, self.HEIGHT, 1]
                        ], dtype=np.float32).T
                        
                        ring_img = H_inv @ ring_world
                        ring_img /= ring_img[2]
                        ring_corners_raw = ring_img[:2].T.astype(np.float32)
                        
                        self.corner_filter.add_sample(ring_corners_raw)
                        
                        h_, w_ = frame_und.shape[:2]
                        xw, yw = self.img_to_world(Hmat, w_/2, h_/2)
                        result['world_pos'] = (xw, yw)
                        result['inside'] = (0 <= xw <= self.WIDTH) and (0 <= yw <= self.HEIGHT)
                        
                        cv2.putText(frame_und,
                                   f"x={xw:.3f}m y={yw:.3f}m {'DENTRO' if result['inside'] else 'FUERA'}",
                                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                   (0, 255, 0) if result['inside'] else (0, 0, 255), 2)
            else:
                missing = []
                if not has_marker_A:
                    missing.append(f"ID {self.ID_A}")
                if not has_marker_B:
                    missing.append(f"ID {self.ID_B}")
                cv2.putText(frame_und, f"Falta marcador: {', '.join(missing)}", 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        ring_corners_median = self.corner_filter.get_median_corners()
        result['corners'] = ring_corners_median
        
        if ring_corners_median is not None:
            poly = ring_corners_median.reshape(-1, 1, 2).astype(np.int32)
            
            if self.corner_filter.is_using_cached_value():
                cv2.polylines(frame_und, [poly], True, (0, 0, 255), 2)
                cv2.putText(frame_und, 
                           f"CACHE - Sin deteccion: {self.corner_filter.frames_without_detection} frames",
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif self.corner_filter.is_ready():
                cv2.polylines(frame_und, [poly], True, (0, 255, 0), 2)
            else:
                cv2.polylines(frame_und, [poly], True, (0, 255, 255), 2)
                cv2.putText(frame_und, 
                           f"Inicializando: {len(self.corner_filter.buffer)}/{self.corner_filter.num_samples}",
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return result


# Si se ejecuta directamente este archivo
if __name__ == "__main__":
    detector = RingDetector()
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        result = detector.process_frame(frame)
        cv2.imshow("Ring Detector", result['frame'])
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

