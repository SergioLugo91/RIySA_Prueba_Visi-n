import math
import cv2
import numpy as np
import sys
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
    M_Base = get_rt_matrix(rvec_base, tvec_base)

    MBase_inv = np.linalg.inv(M_Base)
    M1_in_base = MBase_inv @ M1

    
    tvec_base = M1_in_base[0:3, 3].reshape(3,1)
    rvec_base, _ = cv2.Rodrigues(M1_in_base[0:3, 0:3])
    
    return rvec_base, tvec_base

def get_rt_matrix(R_, T_, force_type=-1):
        """Obtiene la matriz de transformación RT 4x4"""
        R = R_.copy()
        T = T_.copy()

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
        
        print("=" * 60)
        print("PRUEBA TRANSFORMAR EJES AL MARCADOR BASE")
        print("=" * 60)
        # Abrir la cámara PRIMERO
        print("\nAbriendo cámara...")
        cap = cv2.VideoCapture(CAM_ID)
        
        if not cap.isOpened():
            print("ERROR: No se puede abrir la cámara")
            sys.exit(1)
        
        print("✓ Cámara abierta correctamente")
        
        # Verificar que puede capturar
        ret, test_frame = cap.read()
        if not ret:
            print("ERROR: La cámara se abrió pero no puede capturar frames")
            cap.release()
            sys.exit(1)
        
        print(f"✓ Frame de prueba capturado: {test_frame.shape}")

        # Detector ArUco
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

        marker_length=0.048

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

        while 1:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Cargar datos de calibración
            with np.load(CALIBRATION_PATH) as X:
                mtx, dist = X['mtx'], X['dist']

            # Detectar marcadores
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(image)

            if ids is not None:
                for i, c in enumerate(corners):
                    marker_id = int(ids[i][0])
                    center_x = int(np.mean(c[0][:, 0]))
                    center_y = int(np.mean(c[0][:, 1]))

                    # Calcular pose del marcador
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, c,cam_matrix, dist_coeffs
                    )
                    if success:
                        m = Marker(
                            id=marker_id,
                            corners=c,
                            center=(center_x, center_y),
                            rvec=rvec,
                            tvec=tvec
                        )
                        markers.append(m)

                        # Detectar marcador base (IDs 0 o 1). Preferimos 0 sobre 1 si ambos aparecen.
                        if marker_id in (0, 1):
                            if base_marker is None or marker_id == 0:
                                base_marker = m

                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.03)

            if base_marker is not None:
                frame_axes = frame.copy()  # ventana separada
                for m in markers:
                    if m.id != base_marker.id:
                        rvec_trans, tvec_trans = transformar_ejes_a_base(
                            m.rvec, m.tvec, base_marker.rvec, base_marker.tvec
                        )
                        cv2.drawFrameAxes(frame_axes, cam_matrix, dist_coeffs, rvec_trans, tvec_trans, 0.03)

                # Mostrar ventanas separadas
                cv2.imshow("Camara", frame)
                cv2.imshow("Ejes transformados", frame_axes)
        
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