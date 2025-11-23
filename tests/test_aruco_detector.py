import unittest
from src.detectors.aruco_detector import ArUcoDetector  # Asegúrate de que esta función exista en aruco_detector.py

class TestArucoDetector(unittest.TestCase):

    def setUp(self):
        # Configuración inicial para las pruebas
        self.test_image_path = "path/to/test/image.jpg"  # Cambia esto a la ruta de tu imagen de prueba

    def test_marker_detection(self):
        # Prueba la detección de marcadores
        markers, base_marker = ArUcoDetector().detect_markers(self.test_image_path)
        self.assertIsNotNone(markers, "No se detectaron marcadores.")
        self.assertGreater(len(markers), 0, "Se esperaban marcadores, pero no se encontraron.")

    def test_marker_ids(self):
        # Prueba que los IDs de los marcadores sean correctos
        expected_ids = [0, 1]  # Cambia esto según los IDs que esperas detectar
        markers, base_marker = ArUcoDetector().detect_markers(self.test_image_path)
        detected_ids = [marker.id for marker in markers]  # Asegúrate de que esto coincida con tu estructura de datos
        for expected_id in expected_ids:
            self.assertIn(expected_id, detected_ids, f"ID {expected_id} no fue detectado.")

if __name__ == "__main__":
    unittest.main()