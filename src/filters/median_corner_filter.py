import numpy as np
from collections import deque


class MedianCornerFilter:
    """
    Clase que almacena las últimas N muestras de las esquinas de un cuadrado
    y devuelve la mediana de cada coordenada para suavizar el 'tintineo'
    (ruido/jitter) en la detección.
    
    Si no hay detección en un frame, mantiene el último valor conocido
    durante un número limitado de frames.
    """

    def __init__(self, num_samples=10):
        """
        Constructor del filtro.

        Args:
            num_samples (int): Número de muestras que se almacenan
                               para calcular la mediana.
        """

        # Número máximo de muestras a mantener en el buffer
        self.num_samples = num_samples

        # Buffer circular que almacena las últimas N detecciones de esquinas
        # Cada elemento es un array de forma (4, 2)
        self.buffer = deque(maxlen=num_samples)

        # Último valor válido de esquinas detectadas
        self.last_known_corners = None

        # Contador de frames consecutivos sin detección
        self.frames_without_detection = 0

        # Número máximo de frames sin detección antes de invalidar el resultado
        # (aprox. 1 segundo si la cámara va a 30 fps)
        self.max_frames_without_detection = 30
    
    
    def add_sample(self, corners):
        """
        Añade una nueva muestra de esquinas al buffer.

        Args:
            corners (np.ndarray): Array con las 4 esquinas del cuadrado,
                                  forma (4, 2).
                                  Orden esperado: [TL, TR, BR, BL]
        """

        # Comprueba que la muestra sea válida
        if corners is not None and len(corners) == 4:

            # Añade una copia al buffer circular
            self.buffer.append(corners.copy())

            # Actualiza el último valor conocido
            self.last_known_corners = corners.copy()

            # Reinicia el contador de frames sin detección
            self.frames_without_detection = 0
    
    
    def get_median_corners(self):
        """
        Devuelve las esquinas suavizadas mediante la mediana.

        - Si hay muestras recientes: calcula la mediana.
        - Si no hay muestras nuevas: devuelve el último valor conocido.
        - Si se supera el número máximo de frames sin detección: devuelve None.

        Returns:
            np.ndarray: Array de 4 esquinas (4, 2) con valores suavizados,
                        o el último valor conocido,
                        o None si no hay información válida.
        """

        # Si el buffer contiene muestras
        if len(self.buffer) > 0:

            # Convierte el buffer a un array NumPy
            # Forma resultante: (N, 4, 2)
            stacked = np.array(list(self.buffer))

            # Calcula la mediana a lo largo del eje de las muestras
            # Resultado: (4, 2)
            median = np.median(stacked, axis=0)

            # Devuelve la mediana como float32
            return median.astype(np.float32)

        else:
            # No hay muestras nuevas en este frame
            self.frames_without_detection += 1
            
            # Si se supera el límite de frames sin detección,
            # se considera que la detección ya no es válida
            if self.frames_without_detection > self.max_frames_without_detection:
                self.last_known_corners = None
                return None
            
            # Devuelve el último valor conocido (modo "caché")
            return self.last_known_corners
    
    
    def is_ready(self):
        """
        Indica si el filtro ya dispone del número completo de muestras
        para un suavizado estable.

        Returns:
            bool: True si el buffer está lleno.
        """

        return len(self.buffer) == self.num_samples
    
    
    def is_using_cached_value(self):
        """
        Indica si el filtro está devolviendo un valor en caché,
        es decir, no hay detecciones nuevas pero se mantiene
        la última estimación válida.

        Returns:
            bool: True si se está usando el último valor conocido.
        """

        return len(self.buffer) == 0 and self.last_known_corners is not None
    
    
    def reset(self):
        """
        Reinicia completamente el filtro:
        - Vacía el buffer
        - Elimina el último valor conocido
        - Reinicia el contador de frames sin detección
        """

        self.buffer.clear()
        self.last_known_corners = None
        self.frames_without_detection = 0
