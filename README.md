# Sistema de Captura y Segmentación YOLO de Coples

Sistema modular para captura de imágenes con cámara GigE y segmentación de defectos usando modelos YOLO ONNX.

## 🚀 Características

- **Arquitectura Modular**: Código dividido en módulos especializados para fácil mantenimiento
- **Máscaras Pixel-Perfect**: Utiliza coeficientes y prototipos del modelo YOLOv11 para máscaras reales
- **Captura Asíncrona**: Sistema de doble buffer para captura continua sin bloqueos
- **Optimizado para Alta Resolución**: Configurado para imágenes de 1280x1024 píxeles
- **Interfaz Interactiva**: Comandos simples para captura y análisis

## 📁 Estructura del Proyecto

```
Coples/
├── config.py              # Configuraciones y constantes del sistema
├── utils.py               # Funciones auxiliares y utilidades
├── camera_controller.py   # Controlador de cámara GigE
├── image_processor.py     # Procesamiento de imágenes
├── inference_engine.py    # Motor de inferencia ONNX
├── main.py                # Aplicación principal
├── coples_seg_clases.txt  # Archivo de clases del modelo
├── coples_seg1C4V.onnx    # Modelo ONNX de segmentación
└── Salida_cople/          # Directorio de imágenes guardadas
```

## 🏗️ Arquitectura Modular

### `config.py`
Centraliza todas las configuraciones del sistema:
- Parámetros de cámara (exposición, FPS, ROI)
- Configuración de inferencia (umbrales, timeouts)
- Colores y visualización
- Rutas de archivos

### `utils.py`
Funciones auxiliares compartidas:
- Conversión de direcciones IP
- Estadísticas de máscaras
- Filtrado de detecciones
- Gestión de memoria

### `camera_controller.py`
Manejo completo de la cámara GigE:
- Configuración automática
- Captura asíncrona continua
- Sistema de doble buffer
- Estadísticas de rendimiento

### `image_processor.py`
Procesamiento de imágenes:
- Preprocesamiento para el modelo
- Postprocesamiento de máscaras
- Creación de visualizaciones
- Anotaciones en imágenes

### `inference_engine.py`
Motor de inferencia ONNX:
- Carga y configuración del modelo
- Inferencia con timeout
- Procesamiento de máscaras reales
- Manejo de errores

### `main.py`
Aplicación principal:
- Integración de todos los módulos
- Interfaz de usuario
- Gestión de recursos

## 🛠️ Instalación

### Dependencias
```bash
pip install opencv-python numpy onnxruntime
```

### Configuración
1. Asegúrate de tener los archivos del modelo:
   - `coples_seg1C4V.onnx`
   - `coples_seg_clases.txt`
2. Configura la IP de la cámara en `config.py` si es necesario
3. Instala las librerías GigE según tu sistema

## 🎯 Uso del Sistema

### Ejecutar el Sistema
```bash
python main.py
```

### Comandos Disponibles

| Comando | Descripción |
|---------|-------------|
| `ENTER` | Capturar imagen y segmentar con YOLO |
| `v` | Ver frame sin segmentar |
| `s` | Mostrar estadísticas del sistema |
| `c` | Mostrar configuración completa |
| `m` | Cambiar tipo de máscara fallback |
| `q` | Salir del sistema |

### Ejemplo de Uso
```
🎯 Comando: [ENTER]
🔍 RESULTADO DE SEGMENTACIÓN YOLO #1
============================================================
⏱️  TIEMPOS:
   Captura:    2.45 ms
   Inferencia: 150.32 ms
   Total:      155.78 ms

🎯 SEGMENTACIÓN YOLO:
   - Defecto: 1250 píxeles (0.95%)
   - Estado: DEFECTO SEGMENTADO
============================================================
```

## 🔧 Configuración Avanzada

### Parámetros de Cámara
Edita `config.py` para ajustar:
```python
class CameraConfig:
    EXPOSURE_TIME = 15000    # Tiempo de exposición (µs)
    FRAMERATE = 10.0        # FPS
    ROI_WIDTH = 1280        # Ancho del ROI
    ROI_HEIGHT = 1024       # Alto del ROI
```

### Parámetros de Inferencia
```python
class InferenceConfig:
    CONFIDENCE_THRESHOLD = 0.5  # Umbral de confianza
    MAX_DETECTIONS = 3          # Máximo de detecciones
    INFERENCE_TIMEOUT = 2.0     # Timeout de inferencia (s)
```

## 📊 Ventajas del Sistema Modular

### ✅ Mantenibilidad
- Cada módulo tiene una responsabilidad específica
- Fácil localización y corrección de errores
- Código más legible y documentado

### ✅ Escalabilidad
- Fácil agregar nuevas funcionalidades
- Módulos intercambiables
- Pruebas unitarias independientes

### ✅ Reutilización
- Componentes reutilizables en otros proyectos
- Configuración centralizada
- Interfaces bien definidas

### ✅ Desarrollo Paralelo
- Diferentes desarrolladores pueden trabajar en módulos separados
- Menor riesgo de conflictos de código
- Desarrollo más eficiente

## 🔍 Comparación con Versión Anterior

| Aspecto | Versión Anterior | Versión Modular |
|---------|------------------|-----------------|
| **Líneas de código** | 1315 en un archivo | ~1500 en 6 archivos |
| **Mantenimiento** | Difícil | Fácil |
| **Legibilidad** | Compleja | Clara |
| **Pruebas** | Difíciles | Sencillas |
| **Configuración** | Dispersa | Centralizada |
| **Reutilización** | Limitada | Alta |

## 🐛 Solución de Problemas

### Error de Cámara
```
❌ No se encontró la cámara con IP 169.254.8.46
```
**Solución**: Verifica la IP en `config.py` y la conexión de la cámara.

### Error de Modelo
```
❌ No se encontró el modelo: coples_seg1C4V.onnx
```
**Solución**: Asegúrate de que el archivo ONNX esté en el directorio del proyecto.

### Error de Dependencias
```
❌ Dependencias faltantes: pygigev
```
**Solución**: Instala las librerías GigE para tu sistema.

## 🔄 Migración desde Versión Anterior

Para migrar desde `Inferencia_seg_coples.py`:

1. **Configuración**: Ajusta parámetros en `config.py`
2. **Ejecución**: Usa `python main.py` en lugar del archivo anterior
3. **Misma funcionalidad**: Todos los comandos funcionan igual
4. **Mejoras**: Mejor rendimiento y mantenibilidad

## 📈 Rendimiento

El sistema modular mantiene el mismo rendimiento que la versión anterior:
- Captura: ~2-5 ms
- Inferencia: ~100-200 ms (depende del modelo)
- Total: ~150-250 ms por frame

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Identifica el módulo relevante
2. Mantén las interfaces existentes
3. Actualiza `config.py` para nuevos parámetros
4. Documenta los cambios
5. Prueba el sistema completo

## 📄 Licencia

Este proyecto mantiene la misma licencia que el sistema original. 