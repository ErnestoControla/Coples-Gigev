# Sistema de Inferencia de Video para Coples

## 🎬 Descripción

Sistema de inferencia de video en tiempo real para detección de defectos en coples, basado en el modelo `coples_seg1C8V.onnx`. 

### Características Principales:

- **Video en Tiempo Real**: Procesamiento continuo de frames de la cámara GigE
- **Indicador Visual OK/NG**: 
  - ✅ **OK** (Verde): Pieza sin defectos detectados
  - ❌ **NG** (Rojo): Pieza con defectos encontrados
- **Contador de Defectos**: Muestra el número exacto de regiones defectuosas en el frame actual
- **Guardado de Frames**: Los frames individuales se pueden guardar en la carpeta `salida_video/`
- **Grabación de Video**: Grabación continua en formato AVI de toda la sesión de inferencia
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

### Probar Codecs Disponibles

Antes de usar el sistema, puedes verificar qué codecs funcionan en tu sistema:

```bash
python test_codecs.py
```

Este script te dirá cuáles codecs están disponibles y funcionan correctamente.

### Controles Durante la Ejecución

| Tecla | Acción |
|-------|--------|
| `s` | Guardar frame actual con anotaciones |
| `r` | Iniciar/Detener grabación de video |
| `c` | Cambiar codec de video (MJPG/XVID/H264/MP4V) |
| `q` | Salir del sistema |

### Información en Pantalla

El sistema muestra en tiempo real:

1. **Estado Principal**: `OK` (verde) o `NG` (rojo) en la esquina superior izquierda
2. **Contador de Defectos**: Número de regiones defectuosas detectadas
3. **FPS Actual**: Frames por segundo del sistema
4. **Tiempo de Inferencia**: Tiempo que toma el modelo en procesar cada frame
5. **Timestamp**: Hora actual en la esquina inferior derecha
6. **Indicador de Grabación**: Círculo rojo "REC" cuando se está grabando video

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
fps_grabacion = 5.0          # FPS para grabación de video (ajustado al rendimiento real)
```

## 📊 Rendimiento Esperado

- **FPS**: 10-15 FPS (dependiendo de la complejidad de la imagen)
- **Latencia**: ~100-200ms por frame (incluyendo captura + inferencia)
- **Resolución**: 1280x1024 píxeles (ROI de la cámara)
- **Grabación**: Videos AVI a 5 FPS (optimizado para rendimiento real del sistema)

## 🎯 Ejemplos de Salida

### Console Output
```
Frame # 120 | OK | Def:  0 | FPS: 12.3 | Inf:  145.2ms
Frame # 150 | NG | Def:  2 | FPS: 11.8 | Inf:  156.7ms
📸 Frame guardado: cople_video_20250116_143052_123_NG_def2.jpg
```

### Archivos Guardados

**Frames individuales:**
```
salida_video/
├── cople_video_20250116_143052_123_OK_def0.jpg
├── cople_video_20250116_143105_456_NG_def1.jpg
└── cople_video_20250116_143120_789_NG_def3.jpg
```

**Videos grabados:**
```
salida_video/
├── inferencia_cople_20250116_143000.avi
├── inferencia_cople_20250116_144500.avi
└── inferencia_cople_20250116_145200.avi
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

### Video no reproducible / dañado
```
⚠️ Video creado pero muy pequeño: inferencia_cople_xxx.avi (512 bytes)
❌ No se pudo leer ningún frame
```
**Soluciones**:
1. **Cambiar codec durante ejecución**: Presiona `c` para cambiar entre MJPG/XVID/H264/MP4V
2. **Probar codecs disponibles**: Ejecuta `python test_codecs.py` para ver qué funciona
3. **Instalar codecs adicionales**: 
   ```bash
   sudo apt install ffmpeg libx264-dev libxvidcore-dev
   ```
4. **Usar MJPG**: Es el más compatible universalmente

### FPS bajo
**Posibles causas**:
- CPU sobrecargada
- Problemas de red con la cámara
- Modelo de inferencia lento

**Soluciones**:
- Reducir el `framerate` en la configuración
- Aumentar el `confidence_threshold` para procesar menos detecciones
- Cerrar otras aplicaciones pesadas

### Violación de segmento
**Soluciones implementadas**:
- Liberación ordenada de recursos (video → OpenCV → cámara)
- Pausas entre operaciones para evitar conflictos
- Verificación de estado antes de cerrar componentes

## 📝 Notas Técnicas

- El sistema usa **máscaras elípticas** para mejor rendimiento en video
- **Filtrado automático** de contornos pequeños (< 50 píxeles) para reducir ruido
- **Buffers optimizados** (3 buffers) para captura continua sin pérdida de frames
- **Timeout de inferencia** para evitar bloqueos en frames complejos
- **Codec XVID/MJPG** para grabación de video compatible y robusta (fallback automático)
- **Control de FPS dinámico** ajusta la grabación al rendimiento real del sistema 