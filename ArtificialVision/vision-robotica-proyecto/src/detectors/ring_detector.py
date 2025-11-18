import cv2                  
import numpy as np                  
import cv2.aruco as aruco  
from collections import deque
from src.filters.median_corner_filter import MedianCornerFilter

class RingDetector:
    def __init__(self, width=0.80, height=0.80, marker_length=0.09, cam_id=1, target_fps=30.0):
        self.width = width
        self.height = height
        self.marker_length = marker_length
        self.cam_id = cam_id
        self.target_fps = target_fps
        self.corner_filter = MedianCornerFilter(num_samples=10)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.input_video = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        self.cam_matrix, self.dist_coeffs = self.load_calibration_data()

    def load_calibration_data(self):
        data = np.load("calibracion/cam_calib_data.npz")
        cam_matrix = data["K"].astype(np.float32)
        dist_coeffs = data["D"].astype(np.float32)
        return cam_matrix, dist_coeffs

    def detect_rings(self):
        while True:
            ok, frame = self.input_video.read()
            if not ok:
                break

            frame_und = cv2.undistort(frame, self.cam_matrix, self.dist_coeffs)
            gray = cv2.cvtColor(frame_und, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=self.detector_params)

            if ids is not None:
                aruco.drawDetectedMarkers(frame_und, corners, ids)
                self.process_markers(corners, ids)

            cv2.imshow("Detección de Anillos", frame_und)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                break

        self.input_video.release()
        cv2.destroyAllWindows()

    def process_markers(self, corners, ids):
        # Implementar la lógica para procesar los marcadores detectados
        pass

    def world_corners_for_marker_near_00(self, ox, oy, s):
        return np.array([
            [-ox - s, -oy - s],  # TL
            [ox, -oy - s],       # TR
            [ox, -oy],           # BR
            [ox - s, -oy]        # BL
        ], dtype=np.float32)

    def world_corners_for_marker_near_WH(self, ox, oy, s):
        return np.array([
            [self.width + ox, self.height + oy],      # TL
            [self.width + ox + s, self.height + oy],  # TR
            [self.width + ox + s, self.height + oy + s],  # BR
            [self.width + ox, self.height + oy + s]   # BL
        ], dtype=np.float32)

    def img_to_world(self, Hmat, u, v):
        p = np.array([u, v, 1.0], dtype=np.float32)
        q = Hmat @ p
        q /= q[2]
        return float(q[0]), float(q[1])  # x,y en el plano del ring