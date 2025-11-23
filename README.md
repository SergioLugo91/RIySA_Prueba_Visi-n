# Proyecto de Detección de Marcadores y Anillos

Este proyecto está diseñado para implementar un sistema de detección de marcadores ArUco y anillos utilizando visión por computadora. A continuación se describen las principales características y la estructura del proyecto.

## Estructura del Proyecto

```
vision-robotica-proyecto
├── src
│   ├── __init__.py
│   ├── main.py                     # Punto de entrada de la aplicación.
│   ├── detectors                   # Módulo para detectores de marcadores.
│   │   ├── __init__.py
│   │   ├── aruco_detector.py       # Implementación del detector de marcadores ArUco.
│   │   ├── ring_detector.py        # Implementación del detector de anillos.
│   │   └── base_detector.py        # Clase base para detectores.
│   ├── filters                     # Módulo para filtros.
│   │   ├── __init__.py
│   │   └── median_corner_filter.py # Filtro de mediana para suavizar esquinas.
│   ├── utils                       # Módulo de utilidades.
│   │   ├── __init__.py
│   │   ├── camera_calibration.py   # Funciones para calibración de cámara.
│   │   ├── geometry.py             # Funciones geométricas.
│   │   └── transformations.py       # Funciones de transformación de coordenadas.
│   ├── communication               # Módulo para comunicación.
│   │   ├── __init__.py
│   │   └── wifi_sender.py          # Funciones para enviar datos por WiFi.
│   ├── models                      # Módulo para modelos de datos.
│   │   ├── __init__.py
│   │   └── ubot.py                 # Definición de la clase Ubot.
│   └── config                      # Módulo de configuración.
│       ├── __init__.py
│       └── settings.py             # Configuraciones y constantes del proyecto.
├── calibracion                     # Directorio para datos de calibración.
│   └── cam_calib_data.npz         # Datos de calibración de la cámara.
├── tests                           # Directorio para pruebas unitarias.
│   ├── __init__.py
│   ├── test_aruco_detector.py      # Pruebas para el detector de marcadores ArUco.
│   └── test_ring_detector.py       # Pruebas para el detector de anillos.
├── requirements.txt                # Dependencias del proyecto.
├── .gitignore                      # Archivos y directorios a ignorar por Git.
└── README.md                       # Documentación del proyecto.
```

## Instalación

Para instalar las dependencias necesarias, ejecute:

```
pip install -r requirements.txt
```

## Uso

Ejecute el archivo `main.py` para iniciar la aplicación. Este archivo se encarga de importar y ejecutar las funcionalidades de los detectores de marcadores ArUco y anillos.

## Contribuciones

Las contribuciones son bienvenidas. Si desea contribuir, por favor abra un issue o un pull request en el repositorio.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulte el archivo LICENSE para más detalles.