# Punto de entrada de la aplicación para el proyecto de visión robótica

from src.detectors.aruco_detector import detect_aruco_markers
from src.detectors.ring_detector import detect_ring
from src.utils.camera_calibration import load_camera_calibration
from src.communication.wifi_sender import enviarDatos

def main():
    # Cargar matrices de calibración
    camMatrix, distCoeffs = load_camera_calibration("calibracion/cam_calib_data.npz")
    
    # Inicializar video o cámara
    video_source = 0  # Cambiar a la ruta de video si es necesario

    while True:
        # Detectar marcadores ArUco
        aruco_data = detect_aruco_markers(video_source, camMatrix, distCoeffs)
        
        # Detectar anillos
        ring_data = detect_ring(video_source, camMatrix, distCoeffs)
        
        # Enviar datos a través de WiFi
        if aruco_data:
            enviarDatos(aruco_data)
        if ring_data:
            enviarDatos(ring_data)

if __name__ == "__main__":
    main()