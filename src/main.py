# Punto de entrada de la aplicación para el proyecto de visión robótica

import math
import cv2
import numpy as np
import sys
from detectors.ring_detector import RingDetector
from detectors.aruco_detector import ArUcoDetector
from models.ubot import Ubot
from RobPCComm.ComRobotLib.PCComm import RobotComm

def main():
    """
    Sistema integrado de visión artificial para robótica de combate.
    Detecta simultáneamente:
    - Ring de combate (usando marcadores ArUco ID 0 y 1)
    - Robots (cada robot usa 2 marcadores ArUco)
    """
    print("=" * 60)
    print("SISTEMA INTEGRADO DE VISIÓN ARTIFICIAL - ROBÓTICA")
    print("=" * 60)
    print("\nInicializando detectores...")
    
    # Configuración
    CALIBRATION_PATH = "calibracion/cam_calib_data.npz"
    CAM_ID = 1  # ID de la cámara
    TARGET_FPS = 30.0
    
    # CONFIGURACIÓN DE MARCADORES POR ROBOT
    # Cambia estos valores según tus marcadores reales
    ROBOT_MARKERS = {
        0: [2, 3],    # Robot 0 usa ArUco IDs 2 y 3
        1: [4, 5],    # Robot 1 usa ArUco IDs 4 y 5
        2: [6, 7],    # Robot 2 usa ArUco IDs 6 y 7
    }
    
    print("\nConfiguración de marcadores por robot:")
    for robot_id, marker_ids in ROBOT_MARKERS.items():
        print(f"  Robot {robot_id}: ArUcos {marker_ids}")
    
    # Crear detector del ring
    ring_detector = RingDetector(
        width=0.80,
        height=0.80,
        marker_length=0.09,
        id_a=0,
        id_b=1,
        offset_a=(0.0, 0.03),
        offset_b=(0.0, 0.05),
        cam_id=CAM_ID,
        target_fps=TARGET_FPS,
        calibration_path=CALIBRATION_PATH
    )
    
    # Crear detector de robots
    robot_detector = ArUcoDetector(
        marker_length=0.05,
        cam_id=CAM_ID,
        target_fps=TARGET_FPS,
        calibration_path=CALIBRATION_PATH,
        robot_markers=ROBOT_MARKERS
    )
    
    # Configurar comunicación con robots
    RobotCommInstance = RobotComm(logfile="robot_datalog.txt")
    RobotCommInstance.addRobot(0)
    RobotCommInstance.addRobot(1)
    RobotCommInstance.addRobot(2)

    # Usar la misma cámara para ambos detectores
    cap = cv2.VideoCapture(CAM_ID,cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        sys.exit(1)
    
    print("\n✓ Sistema iniciado correctamente")
    print("\nControles:")
    print("  ESC - Salir")
    print("  R - Resetear filtro de esquinas del ring")
    print("  P - Imprimir estado de robots")
    print("-" * 60)
    
    # Variables para el sistema integrado
    frame_count = 0
    datos_robots = {}  # Diccionario {robot_id: Ubot}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nError: No se puede leer el frame de la cámara")
            break
        
        frame_count += 1
        
        # ===== PROCESAR RING =====
        frame_ring, homography, ring_corners, ring_info = ring_detector.process_frame(frame.copy())
        
        # ===== PROCESAR ROBOTS =====
        frame_robots, markers, robot_data, robot_info = robot_detector.process_frame(frame.copy())
        
        # ===== INTEGRAR DATOS =====
        frame_combined = frame.copy()
        h, w = frame_combined.shape[:2]
        
        # Dibujar el ring
        if ring_corners is not None:
            poly = ring_corners.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(frame_combined, [poly], True, (0, 255, 0), 3)
            
            labels = ["(0,0)", "(W,0)", "(W,H)", "(0,H)"]
            for i, corner in enumerate(ring_corners):
                pt = tuple(corner.astype(int))
                cv2.circle(frame_combined, pt, 8, (0, 255, 255), -1)
                cv2.putText(frame_combined, labels[i], 
                           (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Dibujar marcadores individuales
        if len(markers) > 0:
            for marker in markers:
                # Marcadores del ring (0 y 1) - no procesarlos como robots
                if marker.id in [0, 1]:
                    continue
                
                # Dibujar ID del marcador
                cv2.circle(frame_combined, marker.center, 4, (255, 0, 255), -1)
                cv2.putText(frame_combined, f"{marker.id}", 
                           (marker.center[0] + 5, marker.center[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Procesar robots detectados
        for robot_id, data in robot_data.items():
            cx, cy = data['center']
            
            # Dibujar centro del robot
            cv2.circle(frame_combined, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(frame_combined, f"R{robot_id}", 
                       (cx + 15, cy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Si tenemos homografía, verificar si está dentro del ring
            if homography is not None:
                p = np.array([cx, cy, 1.0], dtype=np.float32)
                q = homography @ p
                q /= q[2]
                xw, yw = float(q[0]), float(q[1])
                
                inside = (0 <= xw <= ring_detector.width) and (0 <= yw <= ring_detector.height)
                
                # Inicializar o actualizar Ubot
                if robot_id not in datos_robots:
                    datos_robots[robot_id] = Ubot(
                        id=robot_id,
                        ang=0.0,
                        dist=0.0,
                        Out=1 if not inside else 0
                    )
                else:
                    datos_robots[robot_id].Out = 1 if not inside else 0
                
                color = (0, 255, 0) if inside else (0, 0, 255)
                status = "DENTRO" if inside else "FUERA"
                
                cv2.putText(frame_combined, 
                           f"({xw:.2f},{yw:.2f})m {status}",
                           (cx + 15, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Mostrar distancias y ángulos entre robots
        for pair_info in robot_info['robot_pairs']:
            r1 = pair_info['robot1']
            r2 = pair_info['robot2']
            dist = pair_info['distance']
            ang1 = pair_info['angle1']
            ang2 = pair_info['angle2']
            
            # Actualizar datos de Ubot con distancia y ángulo
            if r1 in datos_robots:
                datos_robots[r1].dist = round(dist,1)
                datos_robots[r1].ang = round(ang1,1)
            
            if r2 in datos_robots:
                datos_robots[r2].dist = round(dist,1)
                datos_robots[r2].ang = round(ang2,1)
            
            # Dibujar línea entre robots
            if r1 in robot_data and r2 in robot_data:
                pt1 = robot_data[r1]['center']
                pt2 = robot_data[r2]['center']
                cv2.line(frame_combined, pt1, pt2, (255, 255, 0), 2)
                
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                cv2.putText(frame_combined, 
                           f"D:{dist*100:.1f}cm A:{ang1:.1f}°",
                           (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
        # Enviar información de los robots
        for robot_id, robot in datos_robots.items():
            RobotCommInstance.enviarRobot(
                id_robot=robot_id,
                ang=robot.ang,
                dist=robot.dist,
                out=robot.Out
            )
            RobotCommInstance.recibirRespuesta()

        
        # ===== INFORMACIÓN EN PANTALLA =====
        y_offset = 30
        
        cv2.putText(frame_combined, f"RING: {len(ring_info['markers_detected'])}/2 marcadores", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        if ring_info['both_markers']:
            cv2.putText(frame_combined, "Ring OK", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame_combined, "Esperando ring...", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        y_offset += 30
        
        num_robots = len(robot_info['robots_detected'])
        cv2.putText(frame_combined, f"ROBOTS: {num_robots}/{len(ROBOT_MARKERS)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Estado de cada robot
        if datos_robots:
            for robot_id, robot in datos_robots.items():
                status_text = "FUERA" if robot.Out == 1 else "DENTRO"
                color = (0, 0, 255) if robot.Out == 1 else (0, 255, 0)
                cv2.putText(frame_combined, 
                           f"R{robot_id}: {status_text} D:{robot.dist*100:.1f}cm A:{robot.ang:.1f}°", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
        
        cv2.putText(frame_combined, f"Frame: {frame_count}", 
                   (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # ===== MOSTRAR VENTANAS =====
        cv2.imshow("Sistema Integrado - Ring + Robots", frame_combined)
        
        # ===== CONTROL DE TECLADO =====
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\nSaliendo del sistema...")
            break
        elif key == 114:  # 'r'
            ring_detector.corner_filter.reset()
            print("Filtro de esquinas reseteado")
        elif key == 112:  # 'p'
            print("\n" + "="*50)
            print("ESTADO DE ROBOTS:")
            for robot_id, robot in datos_robots.items():
                status = "FUERA" if robot.Out == 1 else "DENTRO"
                print(f"  Robot {robot_id}:")
                print(f"    Posición: {status} (Out={robot.Out})")
                print(f"    Distancia: {robot.dist*100:.2f} cm")
                print(f"    Ángulo: {robot.ang:.2f}°")
            print("="*50)
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✓ Sistema finalizado correctamente")
    print("\nEstado final de robots:")
    for robot_id, robot in datos_robots.items():
        status = "FUERA" if robot.Out == 1 else "DENTRO"
        print(f"  Robot {robot_id}: {status}, Dist={robot.dist*100:.1f}cm, Ang={robot.ang:.1f}°")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)