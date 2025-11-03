import numpy as np
import cv2
import glob
import os
import time

#===== Captura de datos ======

CAM_INDEX = 0         
RESOLUTION = (1280,720) # ajusta a tu cámara
SAVE_DIR = "Calibracion/CaptCalibCamSergio"      # carpeta de salida

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAM_INDEX,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
print("Pulsa [SPACE] para guardar, [ESC] para salir.")

i = 0
while True:
    ok, frame = cap.read()
    if not ok: continue
    cv2.imshow("Vista previa (tablero a la vista)", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break            # ESC
    if k == 32:                  # SPACE
        fname = os.path.join(SAVE_DIR, f"img_{i:03d}.png")
        cv2.imwrite(fname, frame)
        print("Guardada:", fname); i += 1; time.sleep(0.2)

cap.release(); cv2.destroyAllWindows()

# === CONFIGURACIÓN DEL TABLERO ===
CHESSBOARD = (7, 7)        # número de esquinas interiores (col x fil)
SQUARE_SIZE = 32.0         # tamaño del cuadro en milímetros (3.3 cm)

# === CARPETA DE IMÁGENES ===
IMG_DIR = "Calibracion/CaptCalib"

# === PREPARACIÓN DE LOS PUNTOS 3D DEL TABLERO (plano Z=0) ===
objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # puntos 3D reales
imgpoints = []  # puntos 2D detectados

# === CARGA DE IMÁGENES ===
images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) + glob.glob(os.path.join(IMG_DIR, "*.png")))

if not images:
    raise FileNotFoundError("No se encontraron imágenes en la carpeta 'calib'.")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# === DETECCIÓN DE ESQUINAS ===
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
        cv2.imshow('Detección de esquinas', img)
        cv2.waitKey(150)
    else:
        print(f"❌ No se detectaron esquinas en {fname}")

cv2.destroyAllWindows()

print(f"\n✅ Esquinas detectadas en {len(objpoints)} de {len(images)} imágenes")

# === CALIBRACIÓN ===
if len(objpoints) < 10:
    raise ValueError("Necesitas al menos 10 imágenes válidas con esquinas detectadas.")

ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== RESULTADOS DE CALIBRACIÓN ===")
print(f"RMS reprojection error: {ret:.4f}")
print("Matriz intrínseca K:")
print(K)
print("Coeficientes de distorsión D:")
print(D.ravel())

# === GUARDA LOS RESULTADOS ===
np.savez("Calibracion/cam_calib_data.npz", K=K, D=D, rms=ret)

# === PRUEBA VISUAL ===
test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
undistorted = cv2.undistort(test_img, K, D, None, newcameramtx)

cv2.imshow('Original', test_img)
cv2.imshow('Corregida', undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
