from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Marker:
    id: int
    corners: np.ndarray
    center: tuple
    rvec: np.ndarray
    tvec: np.ndarray

class ArUcoDetector:
    def __init__(self, marker_length=0.025):
        self.marker_length = marker_length
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector_params = cv2.aruco.DetectorParameters()
    
    def detect_markers(self, image):
        corners, ids, rejected = cv2.aruco.detectMarkers(image, self.dictionary, parameters=self.detector_params)
        markers = []
        
        if ids is not None:
            for i, c in enumerate(corners):
                marker_id = int(ids[i][0])
                center_x = int(np.mean(c[0][:, 0]))
                center_y = int(np.mean(c[0][:, 1]))
                success, rvec, tvec = cv2.solvePnP(self.get_object_points(), c, self.get_camera_matrix(), self.get_dist_coeffs())
                
                if success:
                    markers.append(Marker(id=marker_id, corners=c, center=(center_x, center_y), rvec=rvec, tvec=tvec))
        
        return markers

    def get_object_points(self):
        return np.array([
            [-self.marker_length / 2, self.marker_length / 2, 0],
            [self.marker_length / 2, self.marker_length / 2, 0],
            [self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0]
        ], dtype=np.float32)

    def get_camera_matrix(self):
        data = np.load("calibracion/cam_calib_data.npz")
        return data["K"].astype(np.float32)

    def get_dist_coeffs(self):
        data = np.load("calibracion/cam_calib_data.npz")
        return data["D"].astype(np.float32)