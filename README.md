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
│   ├── detectors/                    # Módulo de detección
│   │   ├── aruco_detector.py         # Detector de marcadores ArUco (poses 3D, robots)
│   │   └── ring_detector.py          # Detector del ring y homografía
│   │
│   ├── filters/                      # Módulo de filtrado
│   │   └── median_corner_filter.py   # Filtro de mediana para suavizar ruido de detección
│   │
│   ├── models/                       # Modelos de datos
│   │   └── ubot.py                   # Clase Ubot (dataclass para info de robots)
│   │
│   ├── RobPCComm/                    # Módulo de comunicación
│   │   └── ComRobotLib/
│   │       ├── PCComm.py             # TCP/IP con robots + Interfaz Flask web
│   │       └── templates/
│   │           └── index.html        # Interfaz web HTML
│   │
│   ├── combat/                       # Ejemplos de combate
│   │
│   ├── example_visión.py             # 🎯 PUNTO DE ENTRADA PRINCIPAL
│   ├── example_transformacion.py     # Ejemplo de transformaciones geométricas
│   └── __init__.py
│
├── Calibracion/                      # Datos y scripts de calibración
│   ├── calibracion.py                # Script para capturar imágenes y calibrar cámara
│   ├── cam_calib_data.npz            # Parámetros intrínsecos de la cámara
│   ├── calibrationSession.mat        # Sesión de calibración MATLAB
│   └── CaptCalib*/                   # Conjuntos de imágenes para calibración
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
    │  example_visión.py (BUCLE PRINCIPAL)│
    ├─────────────────────────────────────┤
    │• Integra datos de detectores        │
    │• Calcula ángulos y distancias       │
    │• Gestiona lógica de combate         │
    └──────────────┬──────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
    ┌──────────────┐    ┌───────────────┐
    │ RobotComm    │    │  Interface    │
    │ (TCP/IP)     │    │  (Flask Web)  │
    ├──────────────┤    ├───────────────┤
    │• Envía datos │    │• Dashboard    │
    │  a robots    │    │  en tiempo    │
    │• Recibe      │    │  real         │
    │  respuestas  │    │• Video stream │
    └──────────────┘    └───────────────┘
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

#### `PCComm.py`
- **Clases**:
  - **`RobotComm`**: Maneja comunicación TCP/IP con robots
    - Envía datos de posición y control
    - Recibe confirmaciones de robots
    - Gestiona múltiples conexiones
  - **`Interface`**: Interfaz web Flask
    - Streaming de video en tiempo real
    - Dashboard de control
    - Monitoreo de estado

### 5. **Punto de Entrada Principal**

#### `example_visión.py` 🎯
- **Función**: `vision_loop()`
  - Bucle principal que procesa frames en tiempo real
  - Integra datos de detectores
  - Calcula ángulos y distancias entre robots
  - Gestiona la lógica de combate
  - Envía datos a robots mediante TCP/IP
  - Actualiza interfaz web con video stream

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
python src/example_visión.py
```

**Interfaz web**: http://localhost:5000

---

## 📊 Flujo de Ejecución

```
1. Inicialización
   ├─ Cargar parámetros de calibración
   ├─ Inicializar ArUcoDetector
   ├─ Inicializar RingDetector
   └─ Inicializar RobotComm e Interface Flask

2. Bucle Principal (vision_loop)
   ├─ Capturar frame de cámara
   ├─ Procesar ring (homografía, esquinas suavizadas)
   ├─ Procesar robots (detectar ArUco, calcular poses 3D)
   ├─ Calcular ángulos y distancias
   ├─ Aplicar lógica de combate
   ├─ Enviar datos a robots (TCP/IP)
   ├─ Actualizar interfaz web
   └─ Mostrar información en pantalla

3. Comunicación Paralela
   ├─ Thread de recepción de datos de robots
   ├─ Thread de servidor web Flask
   └─ Thread principal de visión
```

---

## 🚀 Ejemplo de Uso

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

---

## 📝 Ejemplos Adicionales

- `example_transformacion.py`: Ejemplos de transformaciones geométricas
- `example_transformacion`: Directorio con más ejemplos

---

## 📋 Dependencias Clave

- **OpenCV**: Detección de marcadores ArUco y procesamiento de video
- **NumPy**: Operaciones matriciales y cálculos geométricos
- **Flask**: Interfaz web para monitoreo
- **PySerial**: Comunicación con microcontroladores

---

## 🔍 Notas Técnicas

- **Diccionario ArUco**: DICT_6X6_250 (250 marcadores únicos)
- **Filtrado**: Mediana de 10 muestras para reducir jitter
- **Multithreading**: Separación de visión, comunicación e interfaz web
- **Comunicación**: TCP/IP a puerto personalizado
- **Frecuencia**: 30 FPS objetivo para procesamiento

---

## 📬 Contacto y Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.

---

**Última actualización**: Enero 2026
