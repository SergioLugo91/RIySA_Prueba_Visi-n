import numpy as np

def load_camera_calibration(file_path):
    data = np.load(file_path)
    cam_matrix = data["K"].astype(np.float32)
    dist_coeffs = data["D"].astype(np.float32)
    return cam_matrix, dist_coeffs

def undistort_image(image, cam_matrix, dist_coeffs):
    return cv2.undistort(image, cam_matrix, dist_coeffs)