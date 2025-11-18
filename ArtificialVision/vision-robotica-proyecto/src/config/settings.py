# Configuraciones y constantes del proyecto

# Longitud del marcador ArUco en metros
MARKER_LENGTH = 0.025

# Configuración de la cámara
CAMERA_ID = 0  # ID de la cámara a utilizar
CALIBRATION_DATA_PATH = "calibracion/cam_calib_data.npz"  # Ruta a los datos de calibración

# Configuración de comunicación
UDP_IP = "255.255.255.255"  # Dirección IP para enviar datos
UDP_PORT = 8888  # Puerto para enviar datos

# Configuración de FPS
TARGET_FPS = 30.0  # FPS objetivo para la captura de video
FRAME_PERIOD = 1.0 / TARGET_FPS  # Periodo de fotogramas en segundos

# Configuración de los offsets para los marcadores
OFFSET_A = (0.0, 0.03)  # Offset para el marcador A
OFFSET_B = (0.0, 0.05)  # Offset para el marcador B

# Dimensiones del ring en metros
RING_WIDTH = 0.80
RING_HEIGHT = 0.80