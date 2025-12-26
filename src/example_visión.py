# Punto de entrada de la aplicación para el proyecto de visión robótica

import math
import cv2
import numpy as np
import sys
import time
import threading
from detectors.ring_detector import RingDetector
from detectors.aruco_detector import ArUcoDetector
from models.ubot import Ubot
from RobPCComm.ComRobotLib.PCComm import RobotComm, Interface

# Variables globales para compartir entre threads
datos_ubot = {}
frame_count = 0
TARGET_FPS = 30.0
ROBOT_MARKERS = {}
exit_event = threading.Event() 
ang1_array = []
ang2_array = []

def vision_loop():
    """
    Bucle principal de visión que procesa frames y envía datos a robots.
    """
    global frame_count, datos_ubot, start_time
    
    print("✓ Iniciando bucle de visión...")
    
    while not exit_event.is_set():  # ✅ Verificar evento de salida
        ret, frame = cap.read()
        if not ret:
            print("\nWARNING: No se puede leer el frame de la cámara")
            time.sleep(0.1)
            continue
        
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
                
                inside = (-0.05 <= xw <= ring_detector.width + 0.15) and ( 0.05 <= yw <= ring_detector.height + 0.15 )
                
                # Inicializar o actualizar Ubot
                if robot_id not in datos_ubot:
                    datos_ubot[robot_id] = Ubot(
                        id=robot_id,
                        ang=0.0,
                        dist=0.0,
                        Out=1 if not inside else 0,
                        comm_ok=True
                    )
                else:
                    datos_ubot[robot_id].Out = 1 if not inside else 0
                
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

            # Mediana para suavizar ángulos
            ang1_array.append(pair_info['angle1'])
            ang2_array.append(pair_info['angle2'])
            
            # Actualizar datos de Ubot con distancia y ángulo
            if r1 in datos_ubot:
                datos_ubot[r1].dist = round(dist,2)
                datos_ubot[r1].ang = round(ang1,1)
            
            if r2 in datos_ubot:
                datos_ubot[r2].dist = round(dist,2)
                datos_ubot[r2].ang = round(ang2,1)
            
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
                
        # Envío de datos a robots vía RobotComm cada 5 segundos
        if frame_count % int(TARGET_FPS) == 0:
            if datos_ubot is not None:
                try:
                    time_elapsed = time.time() - start_time
                    print(f"\n[ENVÍO DATOS] Tiempo transcurrido: {time_elapsed:.1f}s")
                    start_time = time.time()
                    ang1 = np.median(ang1_array)
                    std1 = np.std(ang1_array)
                    datos_ubot[r1].ang = round(ang1,1)
                
                    ang2 = np.median(ang2_array)
                    std2 = np.std(ang2_array)
                    datos_ubot[r2].ang = round(ang2,1)
                
                    print(f"[DEBUG] Array ang2 {r2}: {ang2_array}°: size: {len(ang2_array)} : std: {std2:.2f})")
                    print(f"[DEBUG] Array ang1 {r1}: {ang1_array}°: size: {len(ang1_array)} : std: {std1:.2f})")
                    ang1_array.clear()
                    ang2_array.clear()
                    for ubot_id, ubot in datos_ubot.items():
                        RobotCommInstance.enviarRobot(ubot_id,ubot.ang,ubot.dist,ubot.Out)
                        time.sleep(0.2)
                        ubot.comm_ok = RobotCommInstance.recibirRespuesta()
                except Exception as e:
                    print(f"[ERROR] Al enviar datos a robots: {e}")
            else:
                print("\n[ENVÍO DATOS] No hay datos de robots para enviar")

        
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
        if datos_ubot:
            for robot_id, robot in datos_ubot.items():
                status_text = "FUERA" if robot.Out == 1 else "DENTRO"
                comm_text = "OK" if robot.comm_ok else "ERROR"
                color = (0, 0, 255) if robot.Out == 1 else (0, 255, 0)
                cv2.putText(frame_combined, 
                           f"R{robot_id}: {status_text} D:{robot.dist*100:.1f}cm A:{robot.ang:.1f}° Comm:{comm_text}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
        
        cv2.putText(frame_combined, f"Frame: {frame_count}", 
                   (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Actualizar interfaz web con el frame completo
        interface.update_frame(frame_combined)
        
        # ===== MOSTRAR VENTANAS =====
        cv2.imshow("Sistema Integrado - Ring + Robots", frame_combined)
        
        # ===== CONTROL DE TECLADO =====
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\nSaliendo del sistema...")
            exit_event.set()  # ✅ Señalizar salida
            break
        elif key == 114:  # 'r'
            ring_detector.corner_filter.reset()
            print("Filtro de esquinas reseteado")
        elif key == 112:  # 'p'
            print("\n" + "="*50)
            print("ESTADO DE ROBOTS:")
            for robot_id, robot in datos_ubot.items():
                status = "FUERA" if robot.Out == 1 else "DENTRO"
                comm_status = "OK" if robot.comm_ok else "ERROR"
                print(f"  Robot {robot_id}:")
                print(f"    Posición: {status} (Out={robot.Out})")
                print(f"    Distancia: {robot.dist*100:.2f} cm")
                print(f"    Ángulo: {robot.ang:.2f}°")
                print(f"    Comunicación: {comm_status}")
            print("="*50)
        elif key == 13:  # ENTER
            if datos_ubot:
                for pair_info in robot_info['robot_pairs']:
                    r1 = pair_info['robot1']
                    r2 = pair_info['robot2']
                    
                    ang1 = np.median(ang1_array)
                    ang2 = np.median(ang2_array)

                    datos_ubot[r1].ang = round(ang1,1)
                    datos_ubot[r2].ang = round(ang2,1)

                    # Enviar datos
                    for robot_id in [r1, r2]:
                        ang1_array.clear()
                        ang2_array.clear()
                        if robot_id in datos_ubot:
                            ubot = datos_ubot[robot_id]
                            print(f"[MANUAL] Robot {robot_id}: ang={ubot.ang}°, dist={ubot.dist:.1f}cm, out={ubot.Out}")
                            RobotCommInstance.enviarRobot(
                                id_robot=ubot.id,
                                ang=ubot.ang,
                                dist=ubot.dist,
                                out=ubot.Out
                            )
        
        time.sleep(1.0 / TARGET_FPS)  # Limitar a 30 FPS


if __name__ == "__main__":
    try:
        start_time = time.time()
        # Configuración
        CALIBRATION_PATH = "calibracion/cam_calib_data.npz"
        CAM_ID = 0  # ID de la cámara
        TARGET_FPS = 30.0
        
        # CONFIGURACIÓN DE MARCADORES POR ROBOT
        ROBOT_MARKERS = {
            0: [2, 3],    # Robot 0 usa ArUco IDs 2 y 3
            1: [6, 7],    # Robot 1 usa ArUco IDs 4 y 5
            2: [4, 5],    # Robot 2 usa ArUco IDs 6 y 7
        }
        
        print("=" * 60)
        print("SISTEMA INTEGRADO DE VISIÓN ARTIFICIAL - ROBÓTICA")
        print("=" * 60)
        print("\nConfiguración de marcadores por robot:")
        for robot_id, marker_ids in ROBOT_MARKERS.items():
            print(f"  Robot {robot_id}: ArUcos {marker_ids}")
        
        # Abrir la cámara PRIMERO
        print("\nAbriendo cámara...")
        cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
        
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
        
        # Crear detectores
        print("\nInicializando detectores...")
        ring_detector = RingDetector(
            width=0.80,
            height=0.80,
            marker_length=0.14,
            id_a=0,
            id_b=1,
            offset_a=(0.0, 0.03),
            offset_b=(0.0, 0.05),
            cam_id=CAM_ID,
            target_fps=TARGET_FPS,
            calibration_path=CALIBRATION_PATH
        )
        
        robot_detector = ArUcoDetector(
            marker_length=0.076,
            cam_id=CAM_ID,
            target_fps=TARGET_FPS,
            calibration_path=CALIBRATION_PATH,
            robot_markers=ROBOT_MARKERS
        )
        print("✓ Detectores creados")
        
        # Configurar comunicación con robots
        print("\nConfigurando comunicación con robots...")
        RobotCommInstance = RobotComm(ip="192.168.137.161", logfile="robot_datalog.txt")
        RobotCommInstance.addRobot(0)
        RobotCommInstance.addRobot(1)
        RobotCommInstance.addRobot(2)
        print("✓ Comunicación configurada")
        
        # Crear interfaz web
        print("\nCreando interfaz web...")
        interface = Interface(RobotCommInstance)
        print("✓ Interfaz creada")
        
        # Lanzar el bucle de visión en un thread separado
        t_vision = threading.Thread(target=vision_loop, daemon=True)
        t_vision.start()
        print("✓ Thread de visión iniciado")
        
        # Arrancar el servidor Flask en el hilo principal
        print("\n" + "="*60)
        print("Iniciando servidor web en http://localhost:5000")
        print("Presiona ESC en la ventana de visión para detener")
        print("="*60 + "\n")
        
        # ✅ Ejecutar Flask en un thread separado
        t_flask = threading.Thread(target=lambda: interface.run_server(debug=False), daemon=True)
        t_flask.start()
        
        # ✅ Esperar a que exit_event se señalice
        while not exit_event.is_set():
            time.sleep(0.1)
        
        print("\n[SEÑAL] Cerrando programa...")
        
    except KeyboardInterrupt:
        print("\n[CTRL+C] Deteniendo programa...")
        exit_event.set()  
    except Exception as e:
        print(f"\n\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
        exit_event.set()  
    finally:
        # Liberar recursos
        print("\nLiberando recursos...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if RobotCommInstance is not None:
            RobotCommInstance.close()
        
        print("✓ Sistema finalizado correctamente")
        if datos_ubot:
            print("\nEstado final de robots:")
            for robot_id, robot in datos_ubot.items():
                status = "FUERA" if robot.Out == 1 else "DENTRO"
                comm_status = "OK" if robot.comm_ok else "ERROR"
                print(f"  Robot {robot_id}: {status}, Dist={robot.dist:.1f}cm, Ang={robot.ang:.1f}°, Comm={comm_status}")
        print("✓ Recursos liberados")
        sys.exit(0)