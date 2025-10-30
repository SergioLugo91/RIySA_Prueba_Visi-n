import cv2
import numpy as np
import time
import itertools
import math

# --- Configuración inicial ---
video = ""       # ruta de video, o dejar vacío para usar cámara
camId = 0        # ID de cámara (0 = cámara predeterminada)
markerLength = 0.025  # longitud del marcador en metros
estimatePose = True
showRejected = True
target_fps = 30.0
frame_period = 1.0 / target_fps  # ~0.033 segundos

# --- Crear el diccionario y los parámetros del detector ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

# --- Abrir el video o la cámara ---
if video:
    inputVideo = cv2.VideoCapture(video)
    waitTime = 0
else:
    inputVideo = cv2.VideoCapture(camId)
    waitTime = 10

# --- CARGAR MATRICES DE CALIBRACIÓN ---
data = np.load("Calibracion/cam_calib_data.npz")

camMatrix = data["K"].astype(np.float32)
distCoeffs = data["D"].astype(np.float32)

print("Matriz de calibración cargada:")
print(camMatrix)
print("Coeficientes de distorsión:")
print(distCoeffs.ravel())

# --- Definir el sistema de coordenadas del marcador ---
objPoints = np.array([
    [-markerLength/2,  markerLength/2, 0],
    [ markerLength/2,  markerLength/2, 0],
    [ markerLength/2, -markerLength/2, 0],
    [-markerLength/2, -markerLength/2, 0]
], dtype=np.float32)

# --- Función auxiliar: matriz de rotación → ángulos de Euler ---
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])  # roll, pitch, yaw

# --- Control de pares ---
current_pair_index = 0

# --- Bucle principal ---
while True:
    loop_start = time.time()

    ret, image = inputVideo.read()
    if not ret:
        break

    imageCopy = image.copy()
    corners, ids, rejected = detector.detectMarkers(image)

    marker_data = {}  # {id: {"center": (x, y), "yaw": val}}

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(imageCopy, corners, ids)

        for i, c in enumerate(corners):
            marker_id = int(ids[i][0])
            center_x = int(np.mean(c[0][:, 0]))
            center_y = int(np.mean(c[0][:, 1]))

            success, rvec, tvec = cv2.solvePnP(objPoints, c, camMatrix, distCoeffs)
            if not success:
                continue

            # Convertir rotación a yaw
            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = rotationMatrixToEulerAngles(R)

            # Guardar datos
            marker_data[marker_id] = {"center": (center_x, center_y), "yaw": yaw}

            # Dibujar ejes y yaw en la imagen
            cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, markerLength * 1.5)
            cv2.putText(imageCopy, f"yaw={yaw:.1f}", (center_x - 40, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- Agrupar en pares excluyentes ---
    sorted_ids = sorted(marker_data.keys())
    paired_ids = [sorted_ids[i:i+2] for i in range(0, len(sorted_ids), 2)]
    num_pairs = len(paired_ids)

    # --- Dibujar solo el par actual ---
    if 0 <= current_pair_index < num_pairs:
        pair = paired_ids[current_pair_index]
        if len(pair) == 2:
            idA, idB = pair
            pA = marker_data[idA]["center"]
            pB = marker_data[idB]["center"]
            yawA = marker_data[idA]["yaw"]
            yawB = marker_data[idB]["yaw"]

            # Ángulo desde A hacia B
            dx_AB = pB[0] - pA[0]
            dy_AB = pA[1] - pB[1]
            angle_AB = (math.degrees(math.atan2(dx_AB, dy_AB)) + 360) % 360
            delta_A = (angle_AB - yawA + 180) % 360 - 180

            # Ángulo desde B hacia A
            dx_BA = pA[0] - pB[0]
            dy_BA = pB[1] - pA[1]
            angle_BA = (math.degrees(math.atan2(dx_BA, dy_BA)) + 360) % 360
            delta_B = (angle_BA - yawB + 180) % 360 - 180

            mid = ((pA[0]+pB[0])//2, (pA[1]+pB[1])//2)
            cv2.line(imageCopy, pA, pB, (0, 0, 255), 2)
            cv2.putText(imageCopy, f"{idA}-->{idB}: {delta_A:.1f}", (mid[0]-70, mid[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(imageCopy, f"{idB}-->{idA}: {delta_B:.1f}", (mid[0]-70, mid[1]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            print(f"[PAIR {idA}-{idB}] delta_A = {delta_A:.1f}°, delta_B = {delta_B:.1f}°")

    #if showRejected and len(rejected) > 0:
    #    cv2.aruco.drawDetectedMarkers(imageCopy, rejected, borderColor=(100, 0, 255))

    cv2.imshow("Control de Pares (A / D)", imageCopy)

    # --- Control de teclado ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 97:  # a
        current_pair_index = max(0, current_pair_index - 1)
    elif key == 100:  # d
        current_pair_index = min(num_pairs - 1, current_pair_index + 1)
    
    # --- Control de frecuencia (30 Hz) ---
    elapsed = time.time() - loop_start
    sleep_time = frame_period - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

inputVideo.release()
cv2.destroyAllWindows()
