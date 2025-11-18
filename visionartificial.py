import cv2
import numpy as np
from RingDetector import RingDetector
from PruebaArUcos import ArucoRobotDetector

# Clase simple para datos del robot
class Ubot:
    def __init__(self, id, ang=0.0, dist=0.0, Out=0):
        self.id = id
        self.ang = ang
        self.dist = dist
        self.Out = Out  # 0=DENTRO, 1=FUERA

def main():
    print("="*60)
    print("SISTEMA INTEGRADO DE VISIÓN ARTIFICIAL")
    print("="*60)
    
    # Configuración de marcadores por robot
    ROBOT_MARKERS = {
        1: [2, 3],
        2: [4, 5],
        3: [6, 7]
    }
    
    # Crear detectores
    ring_det = RingDetector(width=0.80, height=0.80, marker_len=0.09,
                           id_a=0, id_b=1, ox_a=0.0, oy_a=0.03, ox_b=0.0, oy_b=0.05)
    
    robot_det = ArucoRobotDetector(robot_markers=ROBOT_MARKERS, marker_length=0.025)
    
    # Abrir cámara
    cap = cv2.VideoCapture(0)
    
    print("\nControles:")
    print("  ESC - Salir")
    print("  P - Imprimir estado de robots")
    print("  R - Resetear filtro del ring")
    print("-"*60)
    
    datos_robots = {}
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo cámara")
            break
        
        frame_count += 1
        
        # Procesar ring
        ring_result = ring_det.process_frame(frame.copy())
        
        # Procesar robots
        robot_result = robot_det.process_frame(frame.copy())
        
        # Frame combinado
        combined = frame.copy()
        h, w = combined.shape[:2]
        
        # Dibujar ring
        if ring_result['corners'] is not None:
            poly = ring_result['corners'].reshape(-1, 1, 2).astype(np.int32)
            color = (0, 255, 0) if ring_result['both_detected'] else (0, 165, 255)
            cv2.polylines(combined, [poly], True, color, 3)
        
        # Dibujar marcadores individuales
        for marker_id in robot_result['markers']:
            if marker_id not in [0, 1]:  # Ignorar marcadores del ring
                cv2.putText(combined, f"{marker_id}", 
                           robot_det.detector.detectMarkers(frame)[0][0][0].mean(axis=0).astype(int),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Procesar cada robot
        for robot_id, robot_data in robot_result['robots'].items():
            cx, cy = robot_data['center']
            tvec = robot_data['tvec']
            
            # Dibujar centro del robot
            cv2.circle(combined, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(combined, f"R{robot_id}", (cx + 15, cy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Verificar si está dentro del ring
            if ring_result['homography'] is not None:
                p = np.array([cx, cy, 1.0], dtype=np.float32)
                q = ring_result['homography'] @ p
                q /= q[2]
                xw, yw = float(q[0]), float(q[1])
                
                inside = (0 <= xw <= ring_det.WIDTH) and (0 <= yw <= ring_det.HEIGHT)
                
                # Actualizar Ubot
                if robot_id not in datos_robots:
                    datos_robots[robot_id] = Ubot(id=robot_id, Out=1 if not inside else 0)
                else:
                    datos_robots[robot_id].Out = 1 if not inside else 0
                
                color = (0, 255, 0) if inside else (0, 0, 255)
                status = "DENTRO" if inside else "FUERA"
                cv2.putText(combined, f"{status}", (cx + 15, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Actualizar distancias y ángulos
        for pair in robot_result['pairs']:
            r1, r2 = pair['r1'], pair['r2']
            dist, ang = pair['dist'], pair['ang']
            
            if r1 in datos_robots:
                datos_robots[r1].dist = dist
                datos_robots[r1].ang = ang
            if r2 in datos_robots:
                datos_robots[r2].dist = dist
                datos_robots[r2].ang = ang
            
            # Dibujar línea
            if r1 in robot_result['robots'] and r2 in robot_result['robots']:
                pt1 = robot_result['robots'][r1]['center']
                pt2 = robot_result['robots'][r2]['center']
                cv2.line(combined, pt1, pt2, (255, 255, 0), 2)
                
                mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                cv2.putText(combined, f"D:{dist*100:.1f}cm A:{ang:.1f}°",
                           (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Info en pantalla
        y = 30
        cv2.putText(combined, f"Ring: {'OK' if ring_result['both_detected'] else 'NO'}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
        cv2.putText(combined, f"Robots: {len(robot_result['robots'])}/{len(ROBOT_MARKERS)}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
        
        for rid, robot in datos_robots.items():
            status = "FUERA" if robot.Out else "DENTRO"
            color = (0, 0, 255) if robot.Out else (0, 255, 0)
            cv2.putText(combined, f"R{rid}: {status} D:{robot.dist*100:.1f}cm",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 25
        
        cv2.imshow("Vision Artificial - Ring + Robots", combined)
        
        # Teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):  # P
            print("\n" + "="*50)
            print("ESTADO DE ROBOTS:")
            for rid, robot in datos_robots.items():
                print(f"  Robot {rid}: Out={robot.Out}, Dist={robot.dist*100:.1f}cm, Ang={robot.ang:.1f}°")
            print("="*50)
        elif key == ord('r'):  # R
            ring_det.corner_filter.reset()
            print("Filtro reseteado")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Sistema finalizado")

if __name__ == "__main__":
    main()