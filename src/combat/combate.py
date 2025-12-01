import numpy as np
import time

class CombatController:
    """
    Controlador del combate de robots.
    Gestiona la lógica del combate, detección de robots y estados.
    """
    
    def __init__(self, num_robots=3, ring_width=0.80, ring_height=0.80):
        """
        Inicializa el controlador de combate.
        
        Args:
            num_robots (int): Número de robots en combate (por defecto 2)
            ring_width (float): Ancho del ring en metros
            ring_height (float): Alto del ring en metros
        """
        self.num_robots = num_robots
        self.ring_width = ring_width
        self.ring_height = ring_height
        
        # Estado del combate
        self.combat_active = False
        self.combat_started = False
        self.combat_finished = False
        self.positioning_phase = False  # Fase de posicionamiento inicial
        
        # Control de rounds
        self.current_round = 0
        self.winner_robot_id = None
        self.loser_robot_id = None
        self.waiting_robot_id = None  # Robot que espera su turno (para 3 robots)
        
        # Contadores
        self.frames_both_robots_detected = 0
        self.frames_required_to_start = 30  # Frames necesarios para confirmar detección (1 seg a 30fps)
        self.frames_in_position = 0
        self.frames_required_in_position = 60  # 2 segundos en posición
        
        # Posiciones objetivo para inicio (esquinas opuestas)
        self.start_positions = self._calculate_start_positions()
        
        # Tolerancias
        self.position_tolerance = 0.15  # 15 cm de tolerancia en posición
        self.angle_tolerance = 20.0     # 20 grados de tolerancia en orientación
        
        # Historial
        self.detection_history = []
        self.combat_history = []  # Historial de combates: [{round, winner, loser, duration}]
        
        # Timestamps
        self.positioning_start_time = None
        self.combat_start_time = None
        self.combat_end_time = None
        
        print(f"CombatController: Inicializado para {num_robots} robots")
        print(f"  - Ring: {ring_width}m x {ring_height}m")
        print(f"  - Frames para confirmar detección: {self.frames_required_to_start}")
        print(f"  - Frames para confirmar posición: {self.frames_required_in_position}")
        print(f"  - Posiciones de inicio: {self.start_positions}")
    
    def _calculate_start_positions(self):
        """
        Calcula las posiciones de inicio para cada robot en esquinas opuestas.
        
        Returns:
            dict: {robot_id: {'position': (x, y), 'target_angle': float}}
        """
        if self.num_robots == 2:
            # Robot 1: esquina inferior izquierda, mirando hacia arriba-derecha
            # Robot 2: esquina superior derecha, mirando hacia abajo-izquierda
            return {
                1: {
                    'position': (0.15, 0.15),  # 15cm del borde
                    'target_angle': 45.0       # Mirando hacia el centro
                },
                2: {
                    'position': (self.ring_width - 0.15, self.ring_height - 0.15),
                    'target_angle': -135.0     # Mirando hacia el centro (opuesto)
                }
            }
        else:
            # Para más robots, distribuir en esquinas
            positions = {}
            corners = [
                (0.15, 0.15, 45.0),
                (self.ring_width - 0.15, 0.15, 135.0),
                (self.ring_width - 0.15, self.ring_height - 0.15, -135.0),
                (0.15, self.ring_height - 0.15, -45.0)
            ]
            for i in range(min(self.num_robots, 4)):
                positions[i + 1] = {
                    'position': (corners[i][0], corners[i][1]),
                    'target_angle': corners[i][2]
                }
            return positions
    
    def check_robots_in_view(self, robot_data):
        """
        Verifica que todos los robots necesarios estén detectados en el plano de la cámara.
        
        Args:
            robot_data (dict): Diccionario con datos de robots detectados
                              {robot_id: {'center': (x,y), 'tvec': tvec, 'marker_ids': [id1, id2]}}
        
        Returns:
            dict: Información de detección
        """
        num_detected = len(robot_data)
        all_detected = num_detected >= self.num_robots
        
        detected_ids = set(robot_data.keys())
        expected_ids = set(range(1, self.num_robots + 1))
        missing_robots = list(expected_ids - detected_ids)
        
        if all_detected:
            self.frames_both_robots_detected += 1
        else:
            self.frames_both_robots_detected = 0
        
        ready_to_start = self.frames_both_robots_detected >= self.frames_required_to_start
        
        detection_info = {
            'all_detected': all_detected,
            'num_detected': num_detected,
            'missing_robots': missing_robots,
            'consecutive_frames': self.frames_both_robots_detected,
            'ready_to_start': ready_to_start
        }
        
        self.detection_history.append(detection_info)
        
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
        
        return detection_info
    
    def check_robot_position(self, robot_id, world_position, robot_yaw=None):
        """
        Verifica si un robot está en su posición de inicio correcta.
        
        Args:
            robot_id (int): ID del robot
            world_position (tuple): Posición (x, y) en coordenadas del ring en metros
            robot_yaw (float, optional): Ángulo de orientación del robot en grados
        
        Returns:
            dict: {
                'in_position': bool,
                'distance_to_target': float (metros),
                'angle_error': float (grados, si se proporciona yaw),
                'target_position': tuple
            }
        """
        if robot_id not in self.start_positions:
            return {
                'in_position': False,
                'distance_to_target': float('inf'),
                'angle_error': None,
                'target_position': None
            }
        
        target = self.start_positions[robot_id]
        target_pos = np.array(target['position'])
        current_pos = np.array(world_position)
        
        # Calcular distancia euclidiana
        distance = np.linalg.norm(current_pos - target_pos)
        
        # Verificar posición
        position_ok = distance <= self.position_tolerance
        
        # Verificar orientación si se proporciona
        angle_error = None
        angle_ok = True
        if robot_yaw is not None:
            target_angle = target['target_angle']
            angle_error = abs(self._normalize_angle(robot_yaw - target_angle))
            angle_ok = angle_error <= self.angle_tolerance
        
        return {
            'in_position': position_ok and angle_ok,
            'distance_to_target': distance,
            'angle_error': angle_error,
            'target_position': target['position'],
            'target_angle': target['target_angle']
        }
    
    @staticmethod
    def _normalize_angle(angle):
        """Normaliza un ángulo al rango [-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def check_all_robots_in_position(self, robots_world_data):
        """
        Verifica si TODOS los robots están en sus posiciones de inicio.
        
        Args:
            robots_world_data (dict): {robot_id: {'world_pos': (x,y), 'yaw': float}}
        
        Returns:
            dict: {
                'all_in_position': bool,
                'ready_to_combat': bool,
                'robots_status': {robot_id: position_info},
                'consecutive_frames': int
            }
        """
        robots_status = {}
        all_in_position = True
        
        # Verificar cada robot
        for robot_id in range(1, self.num_robots + 1):
            if robot_id in robots_world_data:
                data = robots_world_data[robot_id]
                position_info = self.check_robot_position(
                    robot_id, 
                    data['world_pos'], 
                    data.get('yaw')
                )
                robots_status[robot_id] = position_info
                
                if not position_info['in_position']:
                    all_in_position = False
            else:
                # Robot no detectado
                robots_status[robot_id] = {
                    'in_position': False,
                    'distance_to_target': float('inf'),
                    'angle_error': None,
                    'target_position': self.start_positions.get(robot_id, {}).get('position')
                }
                all_in_position = False
        
        # Actualizar contador de frames en posición
        if all_in_position:
            self.frames_in_position += 1
        else:
            self.frames_in_position = 0
        
        # Verificar si está listo para combatir
        ready_to_combat = self.frames_in_position >= self.frames_required_in_position
        
        return {
            'all_in_position': all_in_position,
            'ready_to_combat': ready_to_combat,
            'robots_status': robots_status,
            'consecutive_frames': self.frames_in_position
        }
    
    def end_combat(self, loser_robot_id):
        """
        Finaliza el combate actual cuando se detecta un OUT.
        Se llama directamente cuando el sistema de detección de ArUcos 
        confirma que un robot tiene Out=1.
        
        Args:
            loser_robot_id (int): ID del robot que perdió (salió fuera)
        
        Returns:
            dict: Información del combate finalizado
        """
        if not self.combat_active:
            return None
        
        self.combat_active = False
        self.combat_finished = True
        self.combat_end_time = time.time()
        
        # Determinar ganador
        all_robots = set(range(1, self.num_robots + 1))
        active_robots = all_robots - {self.waiting_robot_id} if self.waiting_robot_id else all_robots
        winner_robot_id = list(active_robots - {loser_robot_id})[0]
        
        self.winner_robot_id = winner_robot_id
        self.loser_robot_id = loser_robot_id
        
        # Calcular duración
        combat_duration = self.combat_end_time - self.combat_start_time if self.combat_start_time else 0
        
        # Guardar en historial
        combat_record = {
            'round': self.current_round,
            'winner': winner_robot_id,
            'loser': loser_robot_id,
            'duration': combat_duration,
            'timestamp': self.combat_end_time
        }
        self.combat_history.append(combat_record)
        
        print("\n" + "="*60)
        print("¡COMBATE FINALIZADO!")
        print(f"  Round: {self.current_round}")
        print(f"  Ganador: Robot {winner_robot_id}")
        print(f"  Perdedor: Robot {loser_robot_id} (OUT)")
        print(f"  Duración: {combat_duration:.2f} segundos")
        print("="*60)
        
        return combat_record
    
    def prepare_next_round(self):
        """
        Prepara el siguiente round del combate.
        
        - El ganador vuelve a su posición de inicio original
        - El robot que estaba esperando toma la posición del perdedor
        - El perdedor queda fuera esperando
        
        Returns:
            dict: Información del siguiente round o None si no hay más rounds
        """
        if not self.combat_finished:
            return None
        
        # Incrementar round
        self.current_round += 1
        
        if self.num_robots == 2:
            # Para 2 robots, simplemente reiniciar
            next_round_info = {
                'round': self.current_round,
                'robot1': 1,
                'robot2': 2,
                'waiting': None
            }
        else:
            # Para 3+ robots, rotar posiciones
            # El ganador mantiene su posición
            # El que esperaba entra en la posición del perdedor
            # El perdedor pasa a esperar
            old_waiting = self.waiting_robot_id
            new_waiting = self.loser_robot_id
            entering = old_waiting
            
            # Intercambiar posiciones: el que entra toma la posición del perdedor
            if entering is not None and new_waiting is not None:
                loser_pos = self.start_positions[new_waiting]
                self.start_positions[entering] = loser_pos.copy()
            
            self.waiting_robot_id = new_waiting
            
            next_round_info = {
                'round': self.current_round,
                'winner_stays': self.winner_robot_id,
                'new_challenger': entering,
                'waiting': new_waiting
            }
        
        # Reiniciar estados para nuevo combate
        self.combat_finished = False
        self.combat_started = False
        self.combat_active = False
        self.positioning_phase = False
        self.frames_in_position = 0
        self.winner_robot_id = None
        self.loser_robot_id = None
        
        print("\n" + "="*60)
        print(f"PREPARANDO ROUND {self.current_round}")
        print(f"  Información: {next_round_info}")
        print("="*60)
        
        return next_round_info
    
    def start_positioning_phase(self):
        """
        Inicia la fase de posicionamiento.
        Los robots deben moverse a sus posiciones de inicio.
        """
        if not self.combat_started and not self.positioning_phase:
            self.positioning_phase = True
            self.positioning_start_time = time.time()
            self.frames_in_position = 0
            print("\n" + "="*60)
            print("FASE DE POSICIONAMIENTO INICIADA")
            print(f"Robots deben moverse a posiciones:")
            for robot_id, target in self.start_positions.items():
                if robot_id != self.waiting_robot_id:
                    print(f"  Robot {robot_id}: {target['position']} @ {target['target_angle']}°")
            if self.waiting_robot_id:
                print(f"  Robot {self.waiting_robot_id}: ESPERANDO")
            print("="*60)
            return True
        return False
    
    def start_combat(self):
        """
        Inicia el combate oficial.
        Solo se puede llamar si los robots están en posición.
        """
        if self.positioning_phase and not self.combat_started:
            self.combat_started = True
            self.combat_active = True
            self.combat_start_time = time.time()
            self.positioning_phase = False
            print("\n" + "="*60)
            print(f"¡COMBATE INICIADO! - Round {self.current_round + 1}")
            print("="*60)
            return True
        return False
    
    def get_combat_status_text(self):
        """
        Genera texto con el estado actual del combate.
        
        Returns:
            list: Lista de strings con información
        """
        status_lines = []
        
        if self.combat_active:
            duration = time.time() - self.combat_start_time if self.combat_start_time else 0
            status_lines.append(f"=== COMBATE ACTIVO - Round {self.current_round + 1} ===")
            status_lines.append(f"Tiempo: {duration:.1f}s")
        elif self.combat_finished:
            status_lines.append(f"=== COMBATE FINALIZADO - Round {self.current_round} ===")
            status_lines.append(f"Ganador: Robot {self.winner_robot_id}")
            status_lines.append(f"Perdedor: Robot {self.loser_robot_id}")
        
        return status_lines
    
    def get_positioning_status_text(self, position_info):
        """
        Genera texto para mostrar el estado de posicionamiento.
        
        Args:
            position_info (dict): Retornado por check_all_robots_in_position
        
        Returns:
            list: Lista de strings con información
        """
        status_lines = []
        
        if self.positioning_phase:
            status_lines.append("=== FASE DE POSICIONAMIENTO ===")
            
            for robot_id, status in position_info['robots_status'].items():
                if robot_id == self.waiting_robot_id:
                    continue  # No mostrar robot en espera
                    
                if status['in_position']:
                    status_lines.append(f"Robot {robot_id}: ✓ EN POSICIÓN")
                else:
                    dist = status['distance_to_target']
                    if dist != float('inf'):
                        status_lines.append(f"Robot {robot_id}: ✗ Dist: {dist*100:.1f}cm")
                        if status['angle_error'] is not None:
                            status_lines.append(f"  Ángulo: ±{status['angle_error']:.1f}°")
                    else:
                        status_lines.append(f"Robot {robot_id}: ✗ NO DETECTADO")
            
            if position_info['all_in_position']:
                remaining = self.frames_required_in_position - position_info['consecutive_frames']
                if remaining > 0:
                    status_lines.append(f"Manteniendo posición... {remaining} frames")
                else:
                    status_lines.append("✓✓✓ LISTO PARA COMBATIR ✓✓✓")
        
        return status_lines
    
    def get_detection_status_text(self, detection_info):
        """
        Genera texto descriptivo del estado de detección para mostrar en pantalla.
        
        Args:
            detection_info (dict): Información de detección retornada por check_robots_in_view
        
        Returns:
            list: Lista de strings con información para mostrar
        """
        status_lines = []
        
        if detection_info['all_detected']:
            status_lines.append(f"✓ Todos los robots detectados ({detection_info['num_detected']}/{self.num_robots})")
            
            if detection_info['ready_to_start'] and not self.positioning_phase:
                status_lines.append("✓ LISTO PARA COMENZAR POSICIONAMIENTO")
            elif not self.positioning_phase:
                remaining = self.frames_required_to_start - detection_info['consecutive_frames']
                status_lines.append(f"Esperando estabilidad... {remaining} frames")
        else:
            status_lines.append(f"✗ Robots detectados: {detection_info['num_detected']}/{self.num_robots}")
            
            if detection_info['missing_robots']:
                missing_str = ", ".join([f"R{rid}" for rid in detection_info['missing_robots']])
                status_lines.append(f"Faltan: {missing_str}")
        
        return status_lines
    
    def reset_detection_counter(self):
        """Reinicia el contador de frames consecutivos"""
        self.frames_both_robots_detected = 0
        print("CombatController: Contador de detección reiniciado")
    
    def reset_position_counter(self):
        """Reinicia el contador de frames en posición"""
        self.frames_in_position = 0
        print("CombatController: Contador de posición reiniciado")
    
    def reset(self):
        """Reinicia completamente el controlador de combate"""
        self.combat_active = False
        self.combat_started = False
        self.combat_finished = False
        self.positioning_phase = False
        self.current_round = 0
        self.winner_robot_id = None
        self.loser_robot_id = None
        self.waiting_robot_id = None
        self.frames_both_robots_detected = 0
        self.frames_in_position = 0
        self.positioning_start_time = None
        self.combat_start_time = None
        self.combat_end_time = None
        self.combat_history.clear()
        self.start_positions = self._calculate_start_positions()
        print("CombatController: Sistema completamente reiniciado")
    
    def get_stats(self):
        """
        Obtiene estadísticas del combate.
        
        Returns:
            dict: Estadísticas del combate
        """
        if len(self.detection_history) == 0:
            detection_rate = 0.0
        else:
            successful = sum(1 for d in self.detection_history if d['all_detected'])
            detection_rate = (successful / len(self.detection_history)) * 100
        
        combat_duration = None
        if self.combat_start_time:
            if self.combat_end_time:
                combat_duration = self.combat_end_time - self.combat_start_time
            else:
                combat_duration = time.time() - self.combat_start_time
        
        positioning_duration = None
        if self.positioning_start_time:
            if self.combat_start_time:
                positioning_duration = self.combat_start_time - self.positioning_start_time
            else:
                positioning_duration = time.time() - self.positioning_start_time
        
        return {
            'combat_active': self.combat_active,
            'combat_started': self.combat_started,
            'combat_finished': self.combat_finished,
            'positioning_phase': self.positioning_phase,
            'current_round': self.current_round,
            'total_rounds': len(self.combat_history),
            'detection_rate': detection_rate,
            'consecutive_detections': self.frames_both_robots_detected,
            'consecutive_in_position': self.frames_in_position,
            'history_length': len(self.detection_history),
            'combat_duration_sec': combat_duration,
            'positioning_duration_sec': positioning_duration,
            'combat_history': self.combat_history
        }


# Ejemplo de uso
if __name__ == "__main__":
    controller = CombatController(num_robots=2, ring_width=0.80, ring_height=0.80)
    
    print("\n--- Simulación: Uso directo de Out del diccionario de comunicaciones ---")
    
    # Simular datos de comunicaciones (como vendrían del detector de ArUcos)
    datos_robots = {
        1: {'ang': 45.0, 'dist': 0.5, 'Out': 0},  # Robot 1 DENTRO
        2: {'ang': -135.0, 'dist': 0.5, 'Out': 0}  # Robot 2 DENTRO
    }
    
    # Fase 1: Iniciar combate (asumiendo que ya pasó detección y posicionamiento)
    print("\n1. Iniciando combate...")
    controller.positioning_phase = True  # Simular que pasó por posicionamiento
    controller.start_combat()
    
    # Fase 2: Durante el combate - ambos dentro
    print("\n2. Combate en progreso - ambos robots dentro...")
    for i in range(10):
        # Aquí en el main.py se verificaría: if datos_robots[robot_id]['Out'] == 1
        pass
    
    # Fase 3: Se detecta OUT en el diccionario de comunicaciones
    print("\n3. Sistema de detección reporta: Robot 2 OUT")
    datos_robots[2]['Out'] = 1  # El detector de ArUcos actualiza esto automáticamente
    
    # En main.py se haría:
    # if combat_ctrl.combat_active:
    #     for robot_id, datos in datos_robots.items():
    #         if datos['Out'] == 1:
    #             combat_ctrl.end_combat(robot_id)
    
    controller.end_combat(loser_robot_id=2)
    
    # Fase 4: Preparar siguiente round
    print("\n4. Preparando siguiente round...")
    next_round = controller.prepare_next_round()
    
    print("\n--- Estadísticas ---")
    stats = controller.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")