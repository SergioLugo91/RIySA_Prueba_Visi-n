from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """
    Clase base para detectores de visión. Proporciona una interfaz común
    para los detectores específicos como ArUco y Ring.
    """

    @abstractmethod
    def detect(self, frame):
        """
        Método abstracto para detectar objetos en un frame.
        
        Args:
            frame (np.ndarray): El frame de la imagen en el que se realizará la detección.
        
        Returns:
            list: Lista de objetos detectados.
        """
        pass

    @abstractmethod
    def process_data(self, detected_objects):
        """
        Método abstracto para procesar los datos de los objetos detectados.
        
        Args:
            detected_objects (list): Lista de objetos detectados.
        
        Returns:
            dict: Resultados procesados.
        """
        pass