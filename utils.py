"""
Utilidades auxiliares para el sistema de captura y segmentación de coples
Contiene funciones de propósito general y utilidades compartidas
"""

import os
import numpy as np
from functools import reduce
from config import InferenceConfig, StatsConfig


def ipAddr_from_string(ip_string):
    """
    Convertir dirección IPv4 con puntos a entero.
    
    Args:
        ip_string (str): Dirección IP en formato "192.168.1.1"
        
    Returns:
        int: Dirección IP como entero
    """
    return reduce(lambda a, b: a << 8 | b, map(int, ip_string.split(".")))


def get_segmentation_classes():
    """
    Obtiene las clases de segmentación de coples desde el archivo local.
    Para YOLOv11 segmentación, típicamente hay una clase: 'defecto'
    
    Returns:
        list: Lista de nombres de clases
    """
    classes_file = InferenceConfig.CLASSES_FILE
    
    try:
        if not os.path.exists(classes_file):
            print(f"❌ Error: No se encontró el archivo de clases: {classes_file}")
            print("Por favor, asegúrate de que el archivo 'coples_seg_clases.txt' esté en el directorio")
            print("Para YOLOv11 segmentación, debería contener las clases del modelo")
            return []
        
        # Leer las clases desde el archivo
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"✅ Cargadas {len(classes)} clases de segmentación YOLO de coples")
        return classes
        
    except Exception as e:
        print(f"❌ Error leyendo clases de segmentación de coples: {e}")
        return []


def create_colormap(num_classes):
    """
    Crea un mapa de colores para visualizar la segmentación.
    
    Args:
        num_classes (int): Número de clases
        
    Returns:
        list: Lista de colores RGB
    """
    np.random.seed(StatsConfig.COLOR_SEED)  # Para colores consistentes
    colors = []
    for i in range(num_classes):
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        colors.append(color)
    return colors


def filtrar_detecciones_solapadas(detecciones, iou_threshold=None):
    """
    Filtra detecciones solapadas usando Non-Maximum Suppression simplificado.
    
    Args:
        detecciones (np.ndarray): Array de detecciones [N, 37] o [N, 38]
        iou_threshold (float): Umbral de IoU para filtrar solapamientos
        
    Returns:
        np.ndarray: Detecciones filtradas
    """
    if iou_threshold is None:
        iou_threshold = InferenceConfig.IOU_THRESHOLD
    
    if len(detecciones) <= 1:
        return detecciones
    
    # Ordenar por confianza (mayor a menor)
    indices = np.argsort(detecciones[:, 4])[::-1]
    detecciones_ordenadas = detecciones[indices]
    
    # Lista para mantener detecciones válidas
    detecciones_filtradas = []
    
    for i, det in enumerate(detecciones_ordenadas):
        cx1, cy1, w1, h1 = det[:4]
        
        # Verificar si está demasiado cerca de alguna detección ya aceptada
        es_valida = True
        for det_aceptada in detecciones_filtradas:
            cx2, cy2, w2, h2 = det_aceptada[:4]
            
            # Calcular distancia entre centros
            dist_centros = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # Calcular tamaño promedio
            tamaño_promedio = (w1 + h1 + w2 + h2) / 4
            
            # Si la distancia es menor que el umbral, es solapamiento
            if dist_centros < tamaño_promedio * iou_threshold:
                es_valida = False
                break
        
        if es_valida:
            detecciones_filtradas.append(det)
    
    # Mostrar información sobre el filtrado
    if len(detecciones_filtradas) < len(detecciones):
        eliminadas = len(detecciones) - len(detecciones_filtradas)
        print(f"   🔍 Filtradas {eliminadas} detecciones solapadas")
    
    return np.array(detecciones_filtradas) if detecciones_filtradas else detecciones[:1]


def calcular_estadisticas_mascara(mascara):
    """
    Calcula estadísticas detalladas de una máscara de segmentación.
    
    Args:
        mascara (np.ndarray): Máscara binaria
        
    Returns:
        dict: Diccionario con estadísticas
    """
    import cv2
    
    if mascara is None or not np.any(mascara):
        return {
            'total_pixels': 0,
            'defect_pixels': 0,
            'percentage': 0.0,
            'num_regions': 0,
            'areas': [],
            'avg_area': 0.0,
            'max_area': 0.0
        }
    
    total_pixels = mascara.size
    defect_pixels = np.sum(mascara == 1)
    percentage = (defect_pixels / total_pixels) * 100
    
    # Analizar contornos
    contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_regions = len(contours)
    
    # Calcular áreas
    areas = [cv2.contourArea(cnt) for cnt in contours] if num_regions > 0 else []
    avg_area = np.mean(areas) if areas else 0.0
    max_area = np.max(areas) if areas else 0.0
    
    return {
        'total_pixels': total_pixels,
        'defect_pixels': defect_pixels,
        'percentage': percentage,
        'num_regions': num_regions,
        'areas': areas,
        'avg_area': avg_area,
        'max_area': max_area
    }


def limpiar_memoria():
    """
    Limpia la memoria explícitamente liberando recursos.
    """
    import gc
    gc.collect()


def validar_coordenadas(cx, cy, w_norm, h_norm, img_w, img_h):
    """
    Valida y normaliza coordenadas de bounding box.
    
    Args:
        cx, cy: Coordenadas del centro
        w_norm, h_norm: Ancho y alto normalizados
        img_w, img_h: Dimensiones de la imagen
        
    Returns:
        tuple: (cx, cy, w_norm, h_norm, x1, y1, x2, y2) validadas
    """
    # YOLO v11 segmentación: coordenadas siempre normalizadas (0-1)
    # Forzar normalización si están fuera de rango
    if cx > 1:
        cx = cx / InferenceConfig.INPUT_SIZE  # Normalizar por resolución del modelo
    if cy > 1:
        cy = cy / InferenceConfig.INPUT_SIZE
    if w_norm > 1:
        w_norm = w_norm / InferenceConfig.INPUT_SIZE
    if h_norm > 1:
        h_norm = h_norm / InferenceConfig.INPUT_SIZE
    
    # Convertir coordenadas normalizadas a píxeles
    x_center = int(cx * img_w)
    y_center = int(cy * img_h)
    box_w = int(w_norm * img_w)
    box_h = int(h_norm * img_h)
    
    # Calcular esquinas del bounding box
    x1 = max(0, x_center - box_w // 2)
    y1 = max(0, y_center - box_h // 2)
    x2 = min(img_w, x_center + box_w // 2)
    y2 = min(img_h, y_center + box_h // 2)
    
    return cx, cy, w_norm, h_norm, x1, y1, x2, y2


def generar_nombre_archivo(timestamp, count, extension=None):
    """
    Genera nombre de archivo para guardar imágenes.
    
    Args:
        timestamp (str): Timestamp formateado
        count (int): Número de frame
        extension (str, optional): Extensión del archivo
        
    Returns:
        str: Nombre del archivo
    """
    from config import FileConfig
    
    if extension is None:
        extension = FileConfig.IMAGE_FORMAT
    
    return FileConfig.FILENAME_TEMPLATE.format(
        timestamp=timestamp,
        count=count,
        ext=extension
    )


def log_performance(message, elapsed_time, threshold_ms=100):
    """
    Registra información de rendimiento solo si supera un umbral.
    
    Args:
        message (str): Mensaje descriptivo
        elapsed_time (float): Tiempo transcurrido en ms
        threshold_ms (float): Umbral en ms para mostrar mensaje
    """
    if elapsed_time > threshold_ms:
        print(f"⚠️ {message}: {elapsed_time:.2f} ms (>{threshold_ms}ms)")


def format_time_stats(times_list):
    """
    Formatea estadísticas de tiempos.
    
    Args:
        times_list (list): Lista de tiempos en ms
        
    Returns:
        dict: Estadísticas formateadas
    """
    if not times_list:
        return {'promedio': 0, 'min': 0, 'max': 0, 'std': 0}
    
    times_array = np.array(times_list)
    return {
        'promedio': np.mean(times_array),
        'min': np.min(times_array),
        'max': np.max(times_array),
        'std': np.std(times_array)
    }


def verificar_dependencias():
    """
    Verifica que todas las dependencias estén disponibles.
    
    Returns:
        bool: True si todas las dependencias están disponibles
    """
    dependencias = [
        'cv2',
        'numpy',
        'onnxruntime',
        'pygigev'
    ]
    
    faltantes = []
    for dep in dependencias:
        try:
            __import__(dep)
        except ImportError:
            faltantes.append(dep)
    
    if faltantes:
        print(f"❌ Dependencias faltantes: {', '.join(faltantes)}")
        return False
    
    print("✅ Todas las dependencias están disponibles")
    return True


def mostrar_info_sistema():
    """
    Muestra información del sistema y configuración.
    """
    print("\n" + "="*60)
    print("🎯 SISTEMA DE CAPTURA Y SEGMENTACIÓN YOLO DE COPLES")
    print("="*60)
    print("🚀 IMPLEMENTACIÓN MODULAR CON MÁSCARAS REALES PIXEL-PERFECT")
    print("   - Arquitectura dividida en módulos especializados")
    print("   - Usa coeficientes + prototipos del modelo YOLOv11")
    print("   - Formas exactas del defecto (no aproximaciones)")
    print("   - Porcentajes de área precisos")
    print("   - Fácil mantenimiento y extensión")
    print("="*60) 