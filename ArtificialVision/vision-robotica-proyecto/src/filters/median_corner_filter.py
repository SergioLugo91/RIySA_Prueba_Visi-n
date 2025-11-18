class MedianCornerFilter:
    """
    Clase que almacena las últimas N muestras de las esquinas del cuadrado
    y devuelve la mediana de cada coordenada para suavizar el tintineo.
    Mantiene el último valor conocido cuando no hay detección.
    """
    def __init__(self, num_samples=10):
        """
        Args:
            num_samples (int): Número de muestras a almacenar para el cálculo de la mediana.
        """
        self.num_samples = num_samples
        # Buffer circular para almacenar las esquinas (cada elemento es un array de 4 puntos)
        self.buffer = deque(maxlen=num_samples)
        # Almacena el último valor conocido de las esquinas
        self.last_known_corners = None
        # Contador de frames sin detección
        self.frames_without_detection = 0
        # Máximo de frames sin detección antes de resetear (opcional)
        self.max_frames_without_detection = 30  # ~1 segundo a 30 fps
    
    def add_sample(self, corners):
        """
        Añade una nueva muestra de esquinas al buffer.
        
        Args:
            corners (np.ndarray): Array de 4 puntos (esquinas) con forma (4, 2).
                                  Orden: [TL, TR, BR, BL]
        """
        if corners is not None and len(corners) == 4:
            self.buffer.append(corners.copy())
            self.frames_without_detection = 0  # Resetea el contador
    
    def get_median_corners(self):
        """
        Calcula la mediana de todas las muestras almacenadas.
        Si no hay muestras nuevas, devuelve el último valor conocido.
        
        Returns:
            np.ndarray: Array de 4 puntos (esquinas) con la mediana calculada,
                        o el último valor conocido, o None si nunca hubo detección.
        """
        if len(self.buffer) > 0:
            # Convierte el buffer a un array 3D: (num_muestras, 4_esquinas, 2_coords)
            samples = np.array(list(self.buffer))
            
            # Calcula la mediana a lo largo del eje de las muestras (eje 0)
            median_corners = np.median(samples, axis=0).astype(np.float32)
            
            # Actualiza el último valor conocido
            self.last_known_corners = median_corners.copy()
            
            return median_corners
        else:
            # No hay muestras nuevas, devuelve el último valor conocido
            self.frames_without_detection += 1
            
            # Opcionalmente, resetea si pasa mucho tiempo sin detección
            if self.frames_without_detection > self.max_frames_without_detection:
                self.last_known_corners = None
                
            return self.last_known_corners
    
    def is_ready(self):
        """
        Verifica si el buffer está lleno y listo para usar.
        
        Returns:
            bool: True si el buffer tiene el número completo de muestras.
        """
        return len(self.buffer) == self.num_samples
    
    def is_using_cached_value(self):
        """
        Verifica si se está usando un valor en caché (sin detección nueva).
        
        Returns:
            bool: True si no hay detección nueva y se usa el último valor conocido.
        """
        return len(self.buffer) == 0 and self.last_known_corners is not None
    
    def reset(self):
        """Limpia el buffer de muestras y el último valor conocido."""
        self.buffer.clear()
        self.last_known_corners = None
        self.frames_without_detection = 0