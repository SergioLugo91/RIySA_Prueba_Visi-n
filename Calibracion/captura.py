# capture_chessboard_images.py
import cv2, os, time

CAM_INDEX = 0          
RESOLUTION = (1280,720) # ajusta a tu cámara
SAVE_DIR = "Calibracion/CaptCalibsd"      # carpeta de salida

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
