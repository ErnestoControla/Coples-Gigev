# Sistema de Inferencia de Video para Coples

## 🎬 Descripción

Sistema de inferencia de video en tiempo real para detección de defectos en coples, basado en el modelo `coples_seg1C8V.onnx`. 

### Características Principales:

- **Video en Tiempo Real**: Procesamiento continuo de frames de la cámara GigE
- **Indicador Visual OK/NG**: 
  - ✅ **OK** (Verde): Pieza sin defectos detectados
  - ❌ **NG** (Rojo): Pieza con defectos encontrados
- **Contador de Defectos**: Muestra el número exacto de regiones defectuosas en el frame actual
- **Guardado Automático**: Los frames se pueden guardar en la carpeta `salida_video/`
- **Estadísticas en Tiempo Real**: FPS, tiempo de inferencia, timestamp

## 🚀 Uso del Sistema

### Ejecutar el Sistema

```bash
python ejecutar_video.py
```

O directamente:

```bash
python cople_video/Inferencia_video.py
```

### Controles Durante la Ejecución

| Tecla | Acción |
|-------|--------|
| `s` | Guardar frame actual con anotaciones |
| `q` | Salir del sistema |

### Información en Pantalla

El sistema muestra en tiempo real:

1. **Estado Principal**: `OK` (verde) o `NG` (rojo) en la esquina superior izquierda
2. **Contador de Defectos**: Número de regiones defectuosas detectadas
3. **FPS Actual**: Frames por segundo del sistema
4. **Tiempo de Inferencia**: Tiempo que toma el modelo en procesar cada frame
5. **Timestamp**: Hora actual en la esquina inferior derecha

## 📁 Estructura de Archivos

```
Coples/
├── cople_video/
│   └── Inferencia_video.py     # Sistema principal de video
├── salida_video/               # Frames guardados del video
├── ejecutar_video.py           # Script de ejecución
├── coples_seg1C8V.onnx        # Modelo de segmentación
└── coples_seg_clases.txt      # Archivo de clases
```

## 🔧 Configuración

### Parámetros de la Cámara (en el código)

```python
# Parámetros optimizados para video
exposure_time = 12000    # 12ms - exposición rápida para video fluido
framerate = 15.0         # 15 FPS para video en tiempo real
roi_width = 1280         # Ancho del ROI
roi_height = 1024        # Alto del ROI
roi_offset_x = 1416      # Offset X del ROI
roi_offset_y = 576       # Offset Y del ROI
```

### Parámetros de Inferencia

```python
confidence_threshold = 0.4    # Umbral de confianza más restrictivo para video
max_detections = 5           # Máximo 5 detecciones por frame para rendimiento
inference_timeout = 1.5      # Timeout de 1.5s para inferencia
```

## 📊 Rendimiento Esperado

- **FPS**: 10-15 FPS (dependiendo de la complejidad de la imagen)
- **Latencia**: ~100-200ms por frame (incluyendo captura + inferencia)
- **Resolución**: 1280x1024 píxeles (ROI de la cámara)

## 🎯 Ejemplos de Salida

### Console Output
```
Frame # 120 | OK | Def:  0 | FPS: 12.3 | Inf:  145.2ms
Frame # 150 | NG | Def:  2 | FPS: 11.8 | Inf:  156.7ms
📸 Frame guardado: cople_video_20250116_143052_123_NG_def2.jpg
```

### Archivos Guardados
```
salida_video/
├── cople_video_20250116_143052_123_OK_def0.jpg
├── cople_video_20250116_143105_456_NG_def1.jpg
└── cople_video_20250116_143120_789_NG_def3.jpg
```

## 🔍 Diferencias con el Sistema de Imágenes

| Aspecto | Sistema de Imágenes | Sistema de Video |
|---------|-------------------|-----------------|
| **Modo** | Captura manual (ENTER) | Continuo automático |
| **FPS** | ~10 FPS | 12-15 FPS |
| **Inferencia** | Timeout 2.0s | Timeout 1.5s (más rápido) |
| **Visualización** | Imagen estática | Video fluido |
| **Guardado** | Automático cada captura | Manual con tecla 's' |
| **Optimización** | Precisión máxima | Velocidad + precisión |

## 🐛 Solución de Problemas

### No se encuentra la cámara
```
❗No se encontró la cámara con IP 169.254.8.46
```
**Solución**: Verificar conexión de red y que la cámara esté accesible.

### Modelo no encontrado
```
❌ Error: No se encontró el modelo: ../coples_seg1C8V.onnx
```
**Solución**: Asegurarse de que `coples_seg1C8V.onnx` esté en el directorio raíz.

### FPS bajo
**Posibles causas**:
- CPU sobrecargada
- Problemas de red con la cámara
- Modelo de inferencia lento

**Soluciones**:
- Reducir el `framerate` en la configuración
- Aumentar el `confidence_threshold` para procesar menos detecciones
- Cerrar otras aplicaciones pesadas

## 📝 Notas Técnicas

- El sistema usa **máscaras elípticas** para mejor rendimiento en video
- **Filtrado automático** de contornos pequeños (< 50 píxeles) para reducir ruido
- **Buffers optimizados** (3 buffers) para captura continua sin pérdida de frames
- **Timeout de inferencia** para evitar bloqueos en frames complejos 