# Proyecto de Visión Robótica - Detección de Marcadores ArUco y Ring de Combate

Sistema de visión por computadora para detectar marcadores ArUco, identificar robots y delimitar un ring de combate en aplicaciones de robótica.

---

## 📋 Descripción General

Este proyecto implementa un sistema completo de visión para un entorno de combate robótico que:
- **Detecta marcadores ArUco** para localizar robots en el espacio
- **Calcula poses 3D** de cada robot mediante triangulación de dos marcadores
- **Detecta el ring de combate** usando marcadores de referencia y calcula homografía
- **Comunica datos en tiempo real** a los robots mediante TCP/IP
- **Proporciona interfaz web** para monitoreo y control

---

## 📁 Estructura del Proyecto

```
RIySA_Prueba_Visión/
├── src/                              # Código fuente principal
│   ├── GUI.py                        # 🎯 PUNTO DE ENTRADA PRINCIPAL
│   ├── App.py                        # Clase Interface para interfaz web Flask
│   │
│   ├── detectors/                    # Módulo de detección
│   │   ├── aruco_detector.py         # Detector de marcadores ArUco (poses 3D, robots)
│   │   └── ring_detector.py          # Detector del ring y homografía
│   │
│   ├── filters/                      # Módulo de filtrado
│   │   └── median_corner_filter.py   # Filtro de mediana para suavizar ruido de detección
│   │
│   ├── models/                       # Modelos de datos
│   │   ├── ubot.py                   # Clase Ubot (dataclass para info de robots)
│   │   └── gesture_recognizer.task   # Modelo de MediaPipe para reconocimiento de gestos
│   │
│   ├── utils/                        # Utilidades
│   │   └── hand_control.py           # Control por gestos con MediaPipe
│   │
│   ├── RobPCComm/                    # Módulo de comunicación (submódulo Git)
│   │   └── ComRobotLib/
│   │       ├── PCComm.py             # TCP/IP con robots
│   │       └── ...
│   │
│   ├── templates/                    # Plantillas HTML para interfaz web
│   │   └── index.html                # Interfaz web principal
│   │
│   ├── combat/                       # Ejemplos de combate
│   │
│   ├── example_visión.py             # ⚡ EJEMPLO DE REFERENCIA - Lógica completa de visión
│   ├── example_transformacion.py     # Ejemplo de transformaciones geométricas
│   └── __init__.py
│
├── Calibracion/                      # Datos y scripts de calibración
│   ├── calibracion.py                # Script para capturar imágenes y calibrar cámara
│   ├── cam_calib_data.npz            # Parámetros intrínsecos de la cámara
│   ├── calibrationSession.mat        # Sesión de calibración MATLAB
│   └── CaptCalib*/                   # Conjuntos de imágenes para calibración
│
├── tests/                            # Tests
│   └── test_interface.py             # Tests de la interfaz
│
├── requirements.txt                  # Dependencias del proyecto
├── robot_datalog.txt                 # Log de datos de robots
└── README.md                         # Este archivo

```

---

## 🔗 Jerarquía y Flujo de Datos

```
┌─────────────────────────────────────────────────────────────┐
│         CÁMARA / FLUJO DE VIDEO (OpenCV)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
    ┌──────────────────┐  ┌──────────────────┐
    │ ArUcoDetector    │  │  RingDetector    │
    ├──────────────────┤  ├──────────────────┤
    │• Detecta ArUco   │  │• Detecta ring    │
    │• Calcula poses   │  │• Calcula         │
    │  3D              │  │  homografía      │
    │• Identifica      │  │• Suaviza esquinas│
    │  robots          │  │  (filtro mediana)│
    └────────┬─────────┘  └────────┬─────────┘
             │                     │
             ├─────────────────────┤
             │ MedianCornerFilter  │
             │ (suaviza ruido)     │
             │                     │
             ▼                     ▼
    ┌─────────────────────────────────────┐
    │  GUI.py (BUCLE PRINCIPAL)           │
    ├─────────────────────────────────────┤
    │• Integra datos de detectores        │
    │• Calcula ángulos y distancias       │
    │• Gestiona lógica de combate         │
    │• Reconoce gestos (HandControl)      │
    └──────────────┬──────────────────────┘
                   │
                   ├──────────────┐                    
                   ▼              ▼       
            ┌──────────────┐  ┌──────────────┐
            │ RobotComm    │  │  Interface   │
            │ (TCP/IP)     │  │  (Flask Web) │
            ├──────────────┤  ├──────────────┤
            │• Envía datos │  │• Streaming   │
            │  a robots    │  │• Control web │
            │• Recibe      │  │• Estado de   │
            │  respuestas  │  │  robots      │
            └──────┬───────┘  └──────────────┘
                   │
                   ▼
               ROBOTS FÍSICOS
```

---

## 🎯 Descripción de Módulos Principales

### 1. **Detectores** (`src/detectors/`)

#### `aruco_detector.py`
- **Clase**: `ArUcoDetector`
- **Funcionalidad**:
  - Detecta marcadores ArUco en tiempo real
  - Calcula la pose 3D (posición + rotación) de cada marcador
  - Asocia marcadores a robots (por defecto, 2 marcadores por robot)
  - Localiza el robot a partir de sus dos marcadores
  - Calcula distancia y ángulo entre robots
- **Configuración por defecto**:
  - Robot 0: ArUco IDs 2, 3
  - Robot 1: ArUco IDs 6, 7
  - Robot 2: ArUco IDs 4, 5
- **Parámetros clave**:
  - `marker_length`: Tamaño del marcador (0.076 m)
  - `calibration_path`: Ruta a datos de calibración
  - `target_fps`: FPS objetivo (30.0)

#### `ring_detector.py`
- **Clase**: `RingDetector`
- **Funcionalidad**:
  - Detecta el cuadrilátero del ring usando 2 marcadores de referencia
  - Calcula la homografía (transformación perspectiva)
  - Convierte coordenadas de imagen a coordenadas reales del ring
  - Suaviza el ruido de detección con filtro de mediana
- **Configuración por defecto**:
  - Marcadores de referencia: IDs 0 y 1
  - Dimensiones del ring: 0.80 m × 0.80 m

### 2. **Filtros** (`src/filters/`)

#### `median_corner_filter.py`
- **Clase**: `MedianCornerFilter`
- **Funcionalidad**:
  - Almacena las últimas N detecciones de esquinas
  - Calcula la mediana para reducir ruido/jitter
  - Mantiene el último valor válido durante fallos de detección
- **Parámetros**:
  - `num_samples`: Número de muestras (10 por defecto)
  - `max_frames_without_detection`: Máximo 30 frames (~1 segundo) sin detección

### 3. **Modelos de Datos** (`src/models/`)

#### `ubot.py`
- **Clase**: `Ubot` (dataclass)
- **Atributos**:
  - `id`: Identificador del robot
  - `ang`: Ángulo hacia otro robot
  - `dist`: Distancia al otro robot
  - `Out`: Valor de salida/comando
  - `comm_ok`: Estado de comunicación

### 4. **Comunicación** (`src/RobPCComm/`)

Este módulo se encuentra como submódulo Git externo.

#### `PCComm.py`
- **Clases**:
  - **`RobotComm`**: Maneja comunicación TCP/IP con robots
    - Envía datos de posición y control
    - Recibe confirmaciones de robots
    - Gestiona múltiples conexiones

### 5. **Interfaz Web** (`src/`)

#### `App.py`
- **Clase**: `Interface`
- **Funcionalidad**:
  - Gestiona la interfaz web Flask
  - Proporciona streaming de video en tiempo real
  - Control de inicio/fin de combate desde web
  - Integra reconocimiento de gestos para control
- **Rutas web**:
  - `/`: Página principal con estados de robots
  - `/video_feed`: Stream de video
  - `/start`: Iniciar combate
  - `/stop`: Detener combate

### 6. **Control por Gestos** (`src/utils/`)

#### `hand_control.py`
- **Clase**: `HandControl`
- **Funcionalidad**:
  - Reconocimiento de gestos usando MediaPipe
  - Permite control del sistema mediante gestos de mano
  - Integrado con la interfaz para iniciar combate con gesto "Thumb_Up"

### 7. **Punto de Entrada Principal**

#### `GUI.py` 🎯
- **Función**: `vision_loop()`
  - Bucle principal que procesa frames en tiempo real
  - Integra datos de detectores (ArUco y Ring)
  - Calcula ángulos y distancias entre robots
  - Gestiona la lógica de combate
  - Envía datos a robots mediante TCP/IP
  - Actualiza interfaz web con video stream
  - Reconocimiento de gestos para control
- **Características**:
  - Sistema completo integrado
  - Multithreading para visión y servidor web
  - Control por teclado, web y gestos
  - Logging de estados de robots

### 8. **Ejemplos de Referencia**

#### `example_visión.py` ⚡
- **Código de ejemplo** que demuestra la implementación completa del sistema de visión
- Incluye toda la lógica de:
  - Detección de marcadores ArUco
  - Cálculo de propiedades de robots (posición, ángulo, distancia)
  - Envío de datos a robots
  - Integración de ring y robots
- Útil como referencia para entender la arquitectura del sistema

---

## 🔧 Calibración

### Script de Calibración: `Calibracion/calibracion.py`

**Proceso de calibración**:
1. Captura imágenes del tablero de ajedrez desde múltiples ángulos
2. Detecta esquinas del tablero
3. Calcula parámetros intrínsecos de la cámara
4. Guarda datos en archivo `.npz` (NumPy)

**Uso**:
```bash
cd Calibracion
python calibracion.py
# Presionar SPACE para capturar, ESC para terminar
```

**Archivos generados**:
- `cam_calib_data.npz`: Matriz de calibración (K, distorsión)
- `CaptCalib*/`: Carpetas con imágenes de calibración

---

## ⚙️ Instalación y Uso

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

**Dependencias principales**:
- `opencv-python`: Procesamiento de imágenes
- `numpy`: Operaciones matriciales
- `flask`: Interfaz web
- `serial`: Comunicación con microcontroladores
- `socket`: TCP/IP

### 2. Preparar Calibración
- Asegurar que exista `Calibracion/cam_calib_data.npz`
- Si no existe, ejecutar el script de calibración

### 3. Ejecutar la Aplicación
```bash
python src/GUI.py
```

**Controles disponibles**:
- **Teclado**:
  - `ESC`: Salir del sistema
  - `r`: Resetear filtro de esquinas
  - `p`: Imprimir estado de robots
  - `s`: Iniciar envío automático de datos
  - `ENTER`: Envío manual de datos
- **Web**: http://localhost:5000
  - Botones para iniciar/detener combate
  - Visualización de estados de robots
  - Stream de video en tiempo real
- **Gestos**: 
  - Pulgar arriba (Thumb_Up) para iniciar combate

---

## 📊 Flujo de Ejecución

```
1. Inicialización (GUI.py)
   ├─ Cargar parámetros de calibración
   ├─ Inicializar ArUcoDetector
   ├─ Inicializar RingDetector
   ├─ Inicializar RobotComm
   ├─ Inicializar Interface Flask
   └─ Inicializar HandControl (reconocimiento de gestos)

2. Bucle Principal (vision_loop)
   ├─ Capturar frame de cámara
   ├─ Procesar ring (homografía, esquinas suavizadas)
   ├─ Procesar robots (detectar ArUco, calcular poses 3D)
   ├─ Calcular ángulos y distancias
   ├─ Aplicar lógica de combate
   ├─ Reconocer gestos en el frame
   ├─ Enviar datos a robots (TCP/IP) si está habilitado
   ├─ Actualizar interfaz web
   └─ Mostrar información en pantalla

3. Comunicación Paralela
   ├─ Thread de visión principal
   ├─ Thread de servidor Flask (interfaz web)
   └─ Thread de recepción de datos de robots
```

---

## 🚀 Ejemplo de Uso

### Uso básico desde GUI.py (aplicación principal)

```python
# El archivo GUI.py es el punto de entrada principal
# Ejecutar: python src/GUI.py

# La aplicación inicializa todos los componentes automáticamente:
# - Detectores de ArUco y Ring
# - Comunicación con robots
# - Interfaz web Flask
# - Reconocimiento de gestos
```

### Uso de módulos individuales (para desarrollo/testing)

```python
from detectors.aruco_detector import ArUcoDetector
from detectors.ring_detector import RingDetector

# Crear detectores
robot_detector = ArUcoDetector(
    marker_length=0.076,
    calibration_path="Calibracion/cam_calib_data.npz"
)

ring_detector = RingDetector(
    width=0.80,
    height=0.80,
    marker_length=0.09
)

# Procesar frame
frame_robots, markers, robot_data, _ = robot_detector.process_frame(frame)
frame_ring, homography, ring_corners, _ = ring_detector.process_frame(frame)

# Acceder a datos
for robot_id, robot_pose in robot_data.items():
    print(f"Robot {robot_id}: posición={robot_pose['position']}")
```

### Referencia de implementación completa

Ver `src/example_visión.py` para un ejemplo completo de cómo integrar todos los componentes del sistema de visión, incluyendo:
- Detección de marcadores y robots
- Cálculo de propiedades (ángulos, distancias)
- Envío de datos a robots
- Gestión del ring de combate

---

## 📝 Ejemplos Adicionales

- **`example_visión.py`**: Ejemplo completo de referencia que muestra toda la lógica de visión, cálculo de propiedades de robots y envío de datos. Útil para entender la arquitectura del sistema.
- **`example_transformacion.py`**: Ejemplos de transformaciones geométricas
- **`App.py`**: Puede ejecutarse de forma independiente para probar la interfaz web sin el sistema completo de visión

---

## 📋 Dependencias Clave

- **OpenCV**: Detección de marcadores ArUco y procesamiento de video
- **NumPy**: Operaciones matriciales y cálculos geométricos
- **Flask**: Interfaz web para monitoreo y control
- **PySerial**: Comunicación con microcontroladores
- **MediaPipe**: Reconocimiento de gestos para control por manos

---

## 🔍 Notas Técnicas

- **Diccionario ArUco**: DICT_6X6_250 (250 marcadores únicos)
- **Filtrado**: Mediana de 10 muestras para reducir jitter
- **Multithreading**: Separación de visión, comunicación e interfaz web
- **Comunicación**: TCP/IP a puerto personalizado
- **Frecuencia**: 30 FPS objetivo para procesamiento
- **Control**: Soporta teclado, interfaz web y reconocimiento de gestos
- **Submódulos**: RobPCComm es un submódulo Git externo
---

**Última actualización**: Enero 2026
