import math
import cv2
import numpy as np
import sys
import time
from dataclasses import dataclass

@dataclass
class Marker:
    id: int
    corners: np.ndarray
    center: tuple
    rvec: np.ndarray
    tvec: np.ndarray

def transformar_ejes_a_base(rvec, tvec, rvec_base, tvec_base):
    """Transforma las coordenadas del marcador a un sistema de ejes centrado en el marcador base."""
    M1 = get_rt_matrix(rvec, tvec)
    print("M1:", M1)
    M_Base = get_rt_matrix(rvec_base, tvec_base)
    print("M_Base:", M_Base)

    MBase_inv = np.linalg.inv(M_Base)
    print("MBase_inv:", MBase_inv)
    M1_in_base = MBase_inv @ M1
    print("M1_in_base:", M1_in_base)

    
    tvec_base = M1_in_base[0:3, 3].reshape(3,1)
    print("tvec_base:", tvec_base)
    rvec_base, _ = cv2.Rodrigues(M1_in_base[0:3, 0:3])
    print("rvec_base:", rvec_base)
    
    return rvec_base, tvec_base

def get_rt_matrix(R_, T_, force_type=-1):
        """Obtiene la matriz de transformación RT 4x4"""
        R = R_.copy()
        print("Rvec:", R)
        T = T_.copy()
        print("Tvec:", T)

        Matrix = np.eye(4, dtype=np.float64)
        R33 = Matrix[0:3, 0:3]
        
        if R.size == 3:
            R33[:,:] = cv2.Rodrigues(R)[0]
        elif R.size == 9:
            R33[:,:] = R.astype(np.float64).reshape(3, 3)
        
        for i in range(3):
            Matrix[i, 3] = T.flat[i] if T.ndim > 1 else T[i]
        M = Matrix

        return M

if __name__ == "__main__":
    try:
        # Configuración
        CALIBRATION_PATH = "calibracion/cam_calib_data.npz"
        CAM_ID = 0  # ID de la cámara
        TARGET_FPS = 30.0
        start_time = time.time()
        
        print("=" * 60)
        print("PRUEBA TRANSFORMAR EJES AL MARCADOR BASE")
        print("=" * 60)
        # Abrir la cámara PRIMERO
        print("\nAbriendo cámara...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("ERROR: No se puede abrir la cámara")
            sys.exit(1)
        
        print("✓ Cámara abierta correctamente")

        # Detector ArUco
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

        marker_length=0.055

        obj_points = np.array([
            [-marker_length/2,  marker_length/2, 0],
            [ marker_length/2,  marker_length/2, 0],
            [ marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)


        data = np.load("calibracion/cam_calib_data.npz")
        cam_matrix = data["K"].astype(np.float32)
        dist_coeffs = data["D"].astype(np.float32)
        
        markers = []
        base_marker = None

        print("✓ Esperando estabilización de la cámara...")
        time.sleep(2.0)  # Esperar a que la cámara estabilice
        print("✓ Iniciando detección...\n")

        while True:
            # Capturar frame cada segundo
            if time.time() - start_time >= 0.033:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: No se puede leer frame")
                    break
            
                frame_undist = cv2.undistort(frame, cam_matrix, dist_coeffs)
                gray_undist = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
                # Detectar marcadores
                corners, ids, rejected = detector.detectMarkers(gray_undist)

                if ids is not None:
                    for i, c in enumerate(corners):
                        marker_id = int(ids[i][0])
                        center_x = int(np.mean(c[0][:, 0]))
                        center_y = int(np.mean(c[0][:, 1]))

                        # Calcular pose del marcador
                        success, rvec, tvec = cv2.solvePnP(
                            obj_points, c, cam_matrix, dist_coeffs
                        )
                        print(f"Marker ID {marker_id}: rvec={rvec.flatten()}, tvec={tvec.flatten()}")
                        if success:
                            m = Marker(
                                id=marker_id,
                                corners=c,
                                center=(center_x, center_y),
                                rvec=rvec,
                                tvec=tvec
                            )
                            markers.append(m)

                        cv2.aruco.drawDetectedMarkers(frame_undist, corners, ids)
                        cv2.drawFrameAxes(frame_undist, cam_matrix, dist_coeffs, rvec, tvec, 0.015)
                cv2.imshow("Frame", frame_undist)

                
                rvec_trans, tvec_trans = transformar_ejes_a_base(
                    markers[0].rvec, markers[0].tvec, markers[1].rvec, markers[1].tvec
                )
                    
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC para salir
                    print("ESC presionado. Saliendo...")
                    break
                start_time = time.time()
                continue
        
    except KeyboardInterrupt:
        print("\n[CTRL+C] Deteniendo programa...")
    except Exception as e:
        print(f"\n\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Liberar recursos
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Sistema finalizado correctamente")
        print("✓ Recursos liberados")
        sys.exit(0)