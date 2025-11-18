import pytest
import cv2
import numpy as np
from src.detectors.ring_detector import RingDetector  # Asegúrate de que la clase RingDetector esté implementada en ring_detector.py

@pytest.fixture
def setup_ring_detector():
    # Configuración inicial para el detector de anillos
    detector = RingDetector()
    return detector

def test_ring_detection(setup_ring_detector):
    detector = setup_ring_detector
    # Cargar una imagen de prueba que contenga un anillo
    test_image = cv2.imread('tests/test_images/ring_test_image.jpg')  # Asegúrate de tener esta imagen de prueba

    # Detectar anillos en la imagen
    detected_rings = detector.detect(test_image)

    # Verificar que se detecten anillos
    assert len(detected_rings) > 0, "No se detectaron anillos en la imagen de prueba."

def test_ring_properties(setup_ring_detector):
    detector = setup_ring_detector
    test_image = cv2.imread('tests/test_images/ring_test_image.jpg')  # Asegúrate de tener esta imagen de prueba

    detected_rings = detector.detect(test_image)

    # Verificar propiedades de los anillos detectados
    for ring in detected_rings:
        assert 'center' in ring, "El anillo detectado no tiene una propiedad 'center'."
        assert 'radius' in ring, "El anillo detectado no tiene una propiedad 'radius'."