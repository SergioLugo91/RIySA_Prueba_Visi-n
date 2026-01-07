import numpy as np
import cv2
import glob
import os
import time


# ===================== CAPTURA DE DATOS =====================

# Índice de la cámara (depende del sistema)
CAM_INDEX = 1         
#CAM_INDEX = 0         

# Resolución deseada de captura (ancho, alto)
RESOLUTION = (1280, 720)

# Carpeta donde se guardarán las imágenes de calibración
SAVE_DIR = "Calibracion/CaptCalibCamAlvaro"

# Crea la carpeta si no existe
os.makedirs(SAVE_DIR, exist_ok=True)

# Inicializa la cámara usando DirectShow (Windows)
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

# Configura resolución de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

print("Pulsa [SPACE] para guardar, [ESC] para salir.")

# Contador de imágenes guardadas
i = 0

# Bucle de captura en tiempo real
while True:
    # Captura un frame de la cámara
    ok, frame = cap.read()

    # Si falla la captura, salta a la siguiente iteración
    if not ok:
        continue

    # Muestra la imagen en pantalla
    cv2.imshow("Vista previa (tablero a la vista)", frame)

    # Lee tecla pulsada
    k = cv2.waitKey(1) & 0xFF

    # Tecla ESC → salir
    if k == 27:
        break

    # Tecla SPACE → guardar imagen
    if k == 32:
        # Nombre del archivo con numeración incremental
        fname = os.path.join(SAVE_DIR, f"img_{i:03d}.png")

        # Guarda la imagen capturada
        cv2.imwrite(fname, frame)

        print("Guardada:", fname)

        # Incrementa contador
        i += 1

        # Pequeño retardo para evitar capturas repetidas
        time.sleep(0.2)

# Libera la cámara y cierra ventanas
cap.release()
cv2.destroyAllWindows()


# ===================== CONFIGURACIÓN DEL TABLERO =====================

# Número de esquinas interiores del tablero (columnas, filas)
CHESSBOARD = (7, 7)

# Tamaño real del lado de cada cuadrado (en milímetros)
SQUARE_SIZE = 32.0  # 3.2 cm


# ===================== CARPETA DE IMÁGENES =====================

IMG_DIR = "Calibracion/CaptCalibCamAlvaro3"


# ===================== PUNTOS 3D DEL TABLERO =====================

# Inicializa las coordenadas 3D de las esquinas del tablero (Z = 0)
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)

# Genera una malla de puntos (x, y) para el tablero
objp[:, :2] = np.mgrid[
    0:CHESSBOARD[0],
    0:CHESSBOARD[1]
].T.reshape(-1, 2)

# Escala según el tamaño real del cuadrado
objp *= SQUARE_SIZE

# Lista de puntos 3D reales
objpoints = []

# Lista de puntos 2D detectados en la imagen
imgpoints = []


# ===================== CARGA DE IMÁGENES =====================

# Carga todas las imágenes PNG y JPG de la carpeta
images = sorted(
    glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMG_DIR, "*.png"))
)

# Comprueba que existan imágenes
if not images:
    raise FileNotFoundError(
        "No se encontraron imágenes en la carpeta 'calib'."
    )

# Criterio de parada para la refinación subpíxel
criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ===================== DETECCIÓN DE ESQUINAS =====================

# Recorre todas las imágenes de calibración
for fname in images:
    # Lee la imagen
    img = cv2.imread(fname)

    # Convierte a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecta las esquinas del tablero de ajedrez
    ret, corners = cv2.findChessboardCorners(
        gray, CHESSBOARD, None
    )

    # Si se detectan correctamente
    if ret:
        # Refina la posición de las esquinas a precisión subpíxel
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # Guarda correspondencias 3D ↔ 2D
        objpoints.append(objp)
        imgpoints.append(corners2)

        # Dibuja las esquinas detectadas
        cv2.drawChessboardCorners(
            img, CHESSBOARD, corners2, ret
        )

        # Muestra la imagen
        cv2.imshow('Detección de esquinas', img)
        cv2.waitKey(150)

    else:
        print(f"❌ No se detectaron esquinas en {fname}")

# Cierra ventanas
cv2.destroyAllWindows()

print(f"\n✅ Esquinas detectadas en {len(objpoints)} de {len(images)} imágenes")


# ===================== CALIBRACIÓN DE LA CÁMARA =====================

# Verifica que haya suficientes imágenes válidas
if len(objpoints) < 10:
    raise ValueError(
        "Necesitas al menos 10 imágenes válidas con esquinas detectadas."
    )

# Ejecuta la calibración de cámara
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

# Muestra resultados
print("\n=== RESULTADOS DE CALIBRACIÓN ===")
print(f"RMS reprojection error: {ret:.4f}")
print("Matriz intrínseca K:")
print(K)
print("Coeficientes de distorsión D:")
print(D.ravel())

# === GUARDA LOS RESULTADOS ===
np.savez("Calibracion/cam_calib_data3.npz", K=K, D=D, rms=ret)

# ===================== GUARDADO DE RESULTADOS =====================

# Guarda la calibración en un archivo .npz
np.savez(
    "Calibracion/cam_calib_data.npz",
    K=K,
    D=D,
    rms=ret
)


# ===================== PRUEBA VISUAL =====================

# Carga una imagen de prueba
test_img = cv2.imread(images[0])

# Obtiene dimensiones de la imagen
h, w = test_img.shape[:2]

# Calcula una nueva matriz de cámara optimizada
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    K, D, (w, h), 1, (w, h)
)

# Corrige la distorsión de la imagen
undistorted = cv2.undistort(
    test_img, K, D, None, newcameramtx
)

# Muestra comparación
cv2.imshow('Original', test_img)
cv2.imshow('Corregida', undistorted)

cv2.waitKey(0)
cv2.destroyAllWindows()
