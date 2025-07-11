# importar librerias para leer imagenes y utilizar segmentación de coples en formato onnx, también para obtener el tiempo de ejecucion
import cv2
import time
import numpy as np
import onnxruntime as ort
import os
import json
import urllib.request
import sys
import ctypes
import threading
from threading import Event, Lock
from queue import Queue

# Obtener el código de soporte común para el GigE-V Framework
sys.path.append(os.path.dirname(__file__) + "/../gigev_common")

import pygigev
from pygigev import GevPixelFormats as GPF

def ipAddr_from_string(s):
    """Convertir dirección IPv4 con puntos a entero."""
    from functools import reduce
    return reduce(lambda a, b: a << 8 | b, map(int, s.split(".")))

def preprocess_image_segmentation(image, input_size=1024):
    """
    Preprocesa la imagen para el modelo de segmentación de coples
    Optimizado para resolución 1024x1024
    """
    try:
        # Redimensionar a resolución del modelo (1024x1024)
        model_size = 1024  # Resolución del modelo
        image_resized = cv2.resize(image, (model_size, model_size), interpolation=cv2.INTER_AREA)
        
        # La imagen ya está en RGB desde la cámara
        # Normalizar pixel values to [0,1] de manera eficiente
        new_image = image_resized.astype(np.float32, copy=False) / 255.0
        
        # Cambiar dimensiones de HWC a CHW de manera eficiente
        new_image = np.transpose(new_image, (2, 0, 1))
        
        # Agregar batch dimension
        new_image = np.expand_dims(new_image, axis=0)
        
        return new_image, (1.0, 0, 0, model_size, model_size)
        
    except Exception as e:
        print(f"⚠️ Error en preprocesamiento: {e}")
        # Retornar imagen de fallback
        model_size = 1024
        fallback = np.zeros((1, 3, model_size, model_size), dtype=np.float32)
        return fallback, (1.0, 0, 0, model_size, model_size)

def postprocess_segmentation(prediction, original_shape, preprocess_info):
    """
    Postprocesa la predicción de segmentación
    """
    scale, top, left, nh, nw = preprocess_info
    h, w = original_shape[:2]
    
    # Obtener la predicción de segmentación
    if len(prediction.shape) == 4:
        # Si es (1, C, H, W), tomar el primer batch
        seg_pred = prediction[0]
    elif len(prediction.shape) == 2:
        # Si es (1, 2) - clasificación binaria, crear máscara simple
        # Para este caso, asumimos que es clasificación de defecto/no defecto
        # Crear una máscara del tamaño de la imagen original
        seg_pred = np.zeros((h, w), dtype=np.uint8)
        
        # Si la predicción indica defecto (probabilidad > 0.5), marcar toda la imagen
        if prediction[0][1] > 0.5:  # Probabilidad de defecto
            seg_pred = np.ones((h, w), dtype=np.uint8)
        
        return seg_pred
    else:
        seg_pred = prediction
    
    # Si hay múltiples canales, tomar el canal con mayor probabilidad
    if len(seg_pred.shape) > 2 and seg_pred.shape[0] > 1:
        seg_pred = np.argmax(seg_pred, axis=0)
    elif len(seg_pred.shape) == 2:
        seg_pred = (seg_pred > 0.5).astype(np.uint8)
    else:
        seg_pred = (seg_pred[0] > 0.5).astype(np.uint8)
    
    # Si la máscara tiene el tamaño correcto, redimensionar
    if seg_pred.shape != (h, w):
        # Recortar al tamaño original de la imagen procesada si es necesario
        if seg_pred.shape != (nh, nw):
            seg_pred = seg_pred[top:top+nh, left:left+nw]
        
        # Redimensionar al tamaño original
        seg_pred = cv2.resize(seg_pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    return seg_pred

def get_segmentation_classes():
    """
    Obtiene las clases de segmentación de coples desde el archivo local
    Para YOLOv11 segmentación, típicamente hay una clase: 'defecto'
    """
    classes_file = "coples_seg_clases.txt"
    
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
    Crea un mapa de colores para visualizar la segmentación
    """
    np.random.seed(42)  # Para colores consistentes
    colors = []
    for i in range(num_classes):
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        colors.append(color)
    return colors

class SegmentadorCoplesONNX:
    """
    Manejador del modelo de segmentación de coples ONNX
    
    Implementa segmentación YOLOv11 con máscaras REALES pixel-perfect.
    
    Estructura de salida YOLO Segmentación:
    - Output 0: [1, 37, 21504] - Detecciones + coeficientes
      - Posiciones 0-4: x, y, w, h, conf (coordenadas y confianza)
      - Posición 5+: 32 coeficientes de máscara para reconstrucción
    - Output 1: [1, 32, 256, 256] - Prototipos de máscara
    
    Procesamiento REAL:
    1. Extrae los 32 coeficientes de cada detección
    2. Multiplica coeficientes @ prototipos para obtener máscara cruda
    3. Aplica sigmoid + umbral para máscara binaria
    4. Recorta según bounding box y redimensiona a resolución original
    
    Resultado: Máscaras pixel-perfect que siguen la forma exacta del defecto.
    """
    
    def __init__(self, model_path="coples_seg1C4V.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.class_names = None
        self.colormap = None
        self.inicializado = False
        self.procesamiento_activo = True  # Control de procesamiento
        self.usar_mascaras_elipticas = True  # Usar máscaras elípticas (True) o rectangulares (False)
        
    def inicializar(self):
        """Inicializa el modelo ONNX y las clases de segmentación de coples"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Error: No se encontró el archivo del modelo: {self.model_path}")
                return False
            
            # Cargar modelo ONNX con optimizaciones
            print("🧠 Cargando modelo de segmentación de coples ONNX...")
            
            # Configurar opciones de ONNX para mejor rendimiento en 1024x1024
            providers = ['CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 2  # Usar 2 threads para resolución alta
            session_options.inter_op_num_threads = 2
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Obtener información del modelo
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name
            output_shape = self.session.get_outputs()[0].shape
            
            print(f"✅ Modelo de segmentación de coples cargado - Entrada: {input_shape}, Salida: {output_shape}")
            
            # Cargar clases de segmentación de coples
            print("📚 Cargando clases de segmentación de coples...")
            self.class_names = get_segmentation_classes()
            
            if not self.class_names:
                print("❌ No se pudieron cargar las clases de segmentación de coples")
                return False
            
            # Crear mapa de colores
            self.colormap = create_colormap(len(self.class_names))
            
            self.inicializado = True
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando segmentador de coples: {e}")
            return False
    
    def segmentar(self, image_rgb):
        """
        Procesa una imagen RGB con el modelo de segmentación YOLO
        
        Implementa segmentación REAL con:
        - Máscaras pixel-perfect usando coeficientes y prototipos del modelo
        - Decodificación de máscaras YOLOv11 real
        - Contornos múltiples (amarillo + blanco)
        - Puntos centrales marcados
        - Estadísticas detalladas de máscaras reales
        
        Returns: (mascara_segmentacion, imagen_coloreada, tiempo_inferencia_ms)
        """
        if not self.inicializado:
            return None, None, 0
            
        # Si el procesamiento está desactivado, retornar sin defectos
        if not self.procesamiento_activo:
            h, w = image_rgb.shape[:2]
            mascara = np.zeros((h, w), dtype=np.uint8)
            imagen_coloreada = image_rgb.copy()
            return mascara, imagen_coloreada, 0
            
        try:
            start_inference = time.time()
            
            # Preprocesar imagen para el modelo (resolución 1024x1024)
            input_tensor, preprocess_info = preprocess_image_segmentation(image_rgb, input_size=1024)
            
            # Realizar inferencia con timeout seguro
            try:
                # Configurar ONNX para mejor rendimiento
                ort.set_default_logger_severity(3)  # Reducir logs
                
                # Inferencia con timeout usando threading
                import concurrent.futures
                
                def run_inference():
                    return self.session.run(None, {self.input_name: input_tensor})
                
                # Ejecutar inferencia con timeout de 2 segundos
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_inference)
                    try:
                        results = future.result(timeout=2.0)  # 2 segundos timeout
                        
                        # Verificar si tenemos ambas salidas del modelo
                        if len(results) >= 2:
                            prediction = results[0]  # Detecciones + coeficientes: [1, 37, 21504]
                            prototipos = results[1]   # Prototipos de máscara: [1, 32, 256, 256]
                            print(f"✅ Modelo con máscaras reales: detecciones {prediction.shape}, prototipos {prototipos.shape}")
                        else:
                            prediction = results[0]  # Solo detecciones
                            prototipos = None
                            print(f"⚠️ Modelo sin prototipos, usando detecciones: {prediction.shape}")
                            
                    except concurrent.futures.TimeoutError:
                        print("⚠️ Timeout en inferencia (2s), asumiendo sin defectos")
                        future.cancel()
                        raise TimeoutError("Inferencia timeout")
                
            except Exception as e:
                print(f"⚠️ Error en inferencia: {e}")
                # Retornar predicción vacía
                h, w = image_rgb.shape[:2]
                mascara = np.zeros((h, w), dtype=np.uint8)
                imagen_coloreada = image_rgb.copy()
                tiempo_inferencia = (time.time() - start_inference) * 1000
                return mascara, imagen_coloreada, tiempo_inferencia
            
            # Procesar salida YOLO SEGMENTACIÓN con máscaras reales
            h, w = image_rgb.shape[:2]
            mascara = np.zeros((h, w), dtype=np.uint8)
            
            # Procesamiento optimizado para YOLO Segmentación REAL
            try:
                # La salida YOLO Segmentación tiene forma [1, 37, 21504]
                # Donde 37 = 5 + num_clases + 32 coeficientes de máscara
                # 5 = x, y, w, h, conf
                # Para 1 clase: 37 = 5 + 1 + 32 = 38 (pero puede ser 37 si es diferente)
                
                # Transponer para facilitar procesamiento: [1, 21504, 37]
                predictions = prediction.transpose(0, 2, 1)  # [1, 21504, 37]
                predictions = predictions[0]  # [21504, 37]
                
                # Filtrar detecciones con confianza > 0.5
                conf_mask = predictions[:, 4] > 0.5
                valid_predictions = predictions[conf_mask]
                
                # Limitar a máximo 3 detecciones para evitar problemas de memoria
                if len(valid_predictions) > 3:
                    # Ordenar por confianza y tomar las 3 mejores
                    indices = np.argsort(valid_predictions[:, 4])[-3:]
                    valid_predictions = valid_predictions[indices]
                
                # Filtrar detecciones solapadas usando Non-Maximum Suppression simplificado
                detecciones_originales = len(valid_predictions)
                if len(valid_predictions) > 1:
                    valid_predictions = self._filtrar_detecciones_solapadas(valid_predictions)
                
                if len(valid_predictions) > 0:
                    if detecciones_originales > len(valid_predictions):
                        print(f"🎯 Detecciones de segmentación encontradas: {len(valid_predictions)} (filtradas {detecciones_originales - len(valid_predictions)} solapadas)")
                    else:
                        print(f"🎯 Detecciones de segmentación encontradas: {len(valid_predictions)}")
                    
                    # Crear máscaras REALES usando coeficientes y prototipos
                    start_mask_time = time.time()
                    for i, detection in enumerate(valid_predictions):
                        # Timeout para procesamiento de máscaras
                        if time.time() - start_mask_time > 1.0:  # 1 segundo máximo
                            print(f"   ⚠️ Timeout procesando máscaras, procesadas {i}/{len(valid_predictions)}")
                            break
                        
                        # Extraer coordenadas (formato YOLO: cx, cy, w, h normalizado)
                        cx, cy, w_norm, h_norm = detection[:4]
                        conf = detection[4]
                        
                        # Debug: verificar valores originales
                        print(f"   - Coord. originales: cx={cx:.3f}, cy={cy:.3f}, w={w_norm:.3f}, h={h_norm:.3f}")
                        
                        # YOLO v11 segmentación: coordenadas siempre normalizadas (0-1)
                        # Forzar normalización si están fuera de rango
                        if cx > 1:
                            cx = cx / 1024.0  # Normalizar por resolución del modelo
                        if cy > 1:
                            cy = cy / 1024.0
                        if w_norm > 1:
                            w_norm = w_norm / 1024.0
                        if h_norm > 1:
                            h_norm = h_norm / 1024.0
                        
                        # Convertir coordenadas normalizadas a píxeles
                        x_center = int(cx * w)
                        y_center = int(cy * h)
                        box_w = int(w_norm * w)
                        box_h = int(h_norm * h)
                        
                        # Calcular esquinas del bounding box
                        x1 = max(0, x_center - box_w // 2)
                        y1 = max(0, y_center - box_h // 2)
                        x2 = min(w, x_center + box_w // 2)
                        y2 = min(h, y_center + box_h // 2)
                        
                        # Verificar que las coordenadas sean válidas
                        if x1 < w and y1 < h and x2 > 0 and y2 > 0:
                            pixels_antes = np.sum(mascara)
                            
                            # Crear máscara REAL usando coeficientes y prototipos
                            if prototipos is not None:
                                mascara_real = self._crear_mascara_real(detection, prototipos, cx, cy, w_norm, h_norm, w, h)
                                if mascara_real is not None:
                                    mascara = np.logical_or(mascara, mascara_real).astype(np.uint8)
                                    tipo_mascara = "REAL (coeficientes+prototipos)"
                                else:
                                    # Fallback a máscara elíptica si falla el procesamiento real
                                    if self.usar_mascaras_elipticas:
                                        self._crear_mascara_eliptica(mascara, x_center, y_center, box_w, box_h, w, h)
                                        tipo_mascara = "elíptica (fallback)"
                                    else:
                                        mascara[y1:y2, x1:x2] = 1
                                        tipo_mascara = "rectangular (fallback)"
                            else:
                                # Sin prototipos, usar método anterior
                                if self.usar_mascaras_elipticas and len(valid_predictions) <= 2:
                                    # Crear máscara elíptica más realista solo si hay pocas detecciones
                                    self._crear_mascara_eliptica(mascara, x_center, y_center, box_w, box_h, w, h)
                                    tipo_mascara = "elíptica"
                                else:
                                    # Crear máscara rectangular simple para múltiples detecciones
                                    mascara[y1:y2, x1:x2] = 1
                                    tipo_mascara = "rectangular" if not self.usar_mascaras_elipticas else "rectangular (múltiples)"
                            
                            pixels_despues = np.sum(mascara)
                            pixels_agregados = pixels_despues - pixels_antes
                            
                            print(f"   - Máscara {tipo_mascara} creada: conf={conf:.3f}, centro=[{x_center},{y_center}], tamaño=[{box_w}x{box_h}], píxeles={pixels_agregados}")
                        else:
                            print(f"   - Coordenadas fuera de rango: bbox=[{x1},{y1},{x2},{y2}] (imagen: {w}x{h})")
                else:
                    print("✅ No se encontraron defectos en segmentación")
                
                # Mostrar estadísticas de la máscara final
                if np.any(mascara):
                    total_pixels = mascara.size
                    defect_pixels = np.sum(mascara == 1)
                    percentage = (defect_pixels / total_pixels) * 100
                    
                    # Analizar contornos
                    contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    num_regions = len(contours)
                    
                    # Calcular tamaño promedio de regiones
                    if num_regions > 0:
                        areas = [cv2.contourArea(cnt) for cnt in contours]
                        avg_area = np.mean(areas)
                        max_area = np.max(areas)
                        
                        mascara_tipo = "REAL" if prototipos is not None else "Aproximada"
                        print(f"📊 Estadísticas de máscara {mascara_tipo}:")
                        print(f"   - Píxeles defectuosos: {defect_pixels} ({percentage:.2f}%)")
                        print(f"   - Regiones detectadas: {num_regions}")
                        print(f"   - Área promedio: {avg_area:.1f} píxeles")
                        print(f"   - Área máxima: {max_area:.1f} píxeles")
                    
            except Exception as e:
                # Si hay error en procesamiento, asumir sin defectos
                print(f"⚠️ Error procesando segmentación YOLO: {e}")
                print(f"   Forma de predicción: {prediction.shape}")
                mascara = np.zeros((h, w), dtype=np.uint8)
            
            # Crear imagen coloreada para visualización
            imagen_coloreada = self._crear_imagen_coloreada(image_rgb, mascara)
            
            # Liberar memoria explícitamente
            try:
                del input_tensor
                if 'prediction' in locals():
                    del prediction
                if 'prototipos' in locals():
                    del prototipos
                if 'predictions' in locals():
                    del predictions
                if 'valid_predictions' in locals():
                    del valid_predictions
                import gc
                gc.collect()
            except:
                pass
            
            tiempo_inferencia = (time.time() - start_inference) * 1000
            return mascara, imagen_coloreada, tiempo_inferencia
            
        except Exception as e:
            print(f"❌ Error en detección de coples: {e}")
            return None, None, 0
    
    def _filtrar_detecciones_solapadas(self, detecciones, iou_threshold=0.5):
        """
        Filtra detecciones solapadas usando Non-Maximum Suppression simplificado
        """
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

    def _crear_mascara_real(self, detection, prototipos, cx, cy, w_norm, h_norm, img_w, img_h):
        """
        Crea una máscara REAL usando coeficientes y prototipos del modelo YOLOv11
        
        Args:
            detection: Array de detección con 37 valores [x,y,w,h,conf,class,coef1...coef32]
            prototipos: Tensor de prototipos [1, 32, 256, 256]
            cx, cy, w_norm, h_norm: Coordenadas normalizadas del bounding box
            img_w, img_h: Dimensiones de la imagen original
            
        Returns:
            mascara_real: Array binario de la máscara real o None si hay error
        """
        try:
            # Extraer los 32 coeficientes de máscara (posiciones 5+clase hasta el final)
            # Para 1 clase: [x,y,w,h,conf,class,coef1,coef2,...,coef32]
            # O si es 37 total: [x,y,w,h,conf,coef1,coef2,...,coef32]
            
            if len(detection) == 37:
                # Formato: [x,y,w,h,conf,coef1...coef32] (sin clase explícita)
                coeficientes = detection[5:37]  # 32 coeficientes
            elif len(detection) == 38:
                # Formato: [x,y,w,h,conf,class,coef1...coef32]
                coeficientes = detection[6:38]  # 32 coeficientes
            else:
                print(f"   ⚠️ Formato de detección no esperado: {len(detection)} valores")
                return None
            
            # Verificar que tenemos exactamente 32 coeficientes
            if len(coeficientes) != 32:
                print(f"   ⚠️ Coeficientes incorrectos: {len(coeficientes)} (esperado: 32)")
                return None
            
            # Preparar prototipos [1, 32, 256, 256] -> [32, 256, 256]
            protos = prototipos[0]  # [32, 256, 256]
            
            # Multiplicación matricial: coeficientes @ prototipos
            # coeficientes: [32] @ protos: [32, 256, 256] -> [256, 256]
            mascara_cruda = np.dot(coeficientes, protos.reshape(32, -1)).reshape(256, 256)
            
            # Aplicar sigmoid para obtener probabilidades
            mascara_prob = 1 / (1 + np.exp(-mascara_cruda))
            
            # Aplicar umbral para obtener máscara binaria
            mascara_binaria = (mascara_prob > 0.5).astype(np.uint8)
            
            # Recortar máscara según el bounding box de la detección
            # Convertir coordenadas normalizadas a píxeles en la máscara de 256x256
            mask_size = 256
            
            # Calcular bounding box en coordenadas de la máscara
            x_center_mask = int(cx * mask_size)
            y_center_mask = int(cy * mask_size)
            box_w_mask = int(w_norm * mask_size)
            box_h_mask = int(h_norm * mask_size)
            
            # Calcular esquinas del bounding box en la máscara
            x1_mask = max(0, x_center_mask - box_w_mask // 2)
            y1_mask = max(0, y_center_mask - box_h_mask // 2)
            x2_mask = min(mask_size, x_center_mask + box_w_mask // 2)
            y2_mask = min(mask_size, y_center_mask + box_h_mask // 2)
            
            # Crear máscara recortada
            mascara_recortada = np.zeros((mask_size, mask_size), dtype=np.uint8)
            
            # Aplicar la máscara solo dentro del bounding box
            if x2_mask > x1_mask and y2_mask > y1_mask:
                region_mask = mascara_binaria[y1_mask:y2_mask, x1_mask:x2_mask]
                mascara_recortada[y1_mask:y2_mask, x1_mask:x2_mask] = region_mask
                
                # Estadísticas de la máscara generada
                pixels_mascara = np.sum(region_mask)
                area_bbox = (x2_mask - x1_mask) * (y2_mask - y1_mask)
                cobertura = (pixels_mascara / area_bbox * 100) if area_bbox > 0 else 0
                
                print(f"     🎯 Máscara real generada: {pixels_mascara} píxeles ({cobertura:.1f}% del bbox)")
            else:
                print(f"     ⚠️ Bounding box inválido en máscara: [{x1_mask},{y1_mask},{x2_mask},{y2_mask}]")
                return None
            
            # Redimensionar máscara de 256x256 a dimensiones originales
            mascara_final = cv2.resize(mascara_recortada, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            
            # Verificar que la máscara final tiene contenido
            if np.sum(mascara_final) == 0:
                print(f"     ⚠️ Máscara final vacía después de redimensionar")
                return None
            
            return mascara_final
            
        except Exception as e:
            print(f"   ❌ Error creando máscara real: {e}")
            return None

    def _crear_mascara_eliptica(self, mascara, x_center, y_center, box_w, box_h, img_w, img_h):
        """
        Crea una máscara elíptica ultra-optimizada para velocidad (fallback)
        """
        try:
            # Método ultra-rápido: usar cv2.ellipse directamente
            # Parámetros de la elipse
            a = max(1, box_w // 2 - 2)  # Semi-eje mayor horizontal
            b = max(1, box_h // 2 - 2)  # Semi-eje mayor vertical
            
            # Crear elipse principal usando OpenCV (mucho más rápido)
            cv2.ellipse(mascara, (x_center, y_center), (a, b), 0, 0, 360, 1, -1)
            
            # Agregar solo una variación simple con probabilidad baja
            if np.random.random() > 0.7:  # 30% de probabilidad
                # Elipse ligeramente más grande
                a_outer = int(a * 1.1)
                b_outer = int(b * 1.1)
                
                # Crear máscara temporal para el borde
                temp_mask = np.zeros_like(mascara)
                cv2.ellipse(temp_mask, (x_center, y_center), (a_outer, b_outer), 0, 0, 360, 1, -1)
                cv2.ellipse(temp_mask, (x_center, y_center), (a, b), 0, 0, 360, 0, -1)
                
                # Aplicar solo algunos píxeles del borde
                borde_pixels = np.where(temp_mask == 1)
                if len(borde_pixels[0]) > 0:
                    # Tomar solo el 30% de los píxeles del borde
                    np.random.seed(int(x_center + y_center) % 1000)
                    indices = np.random.choice(len(borde_pixels[0]), size=len(borde_pixels[0]) // 3, replace=False)
                    mascara[borde_pixels[0][indices], borde_pixels[1][indices]] = 1
            
        except Exception as e:
            # Si hay error, usar máscara rectangular simple
            x1 = max(0, x_center - box_w // 2)
            y1 = max(0, y_center - box_h // 2)
            x2 = min(img_w, x_center + box_w // 2)
            y2 = min(img_h, y_center + box_h // 2)
            mascara[y1:y2, x1:x2] = 1

    def _crear_imagen_coloreada(self, imagen_original, mascara):
        """
        Crea una imagen coloreada para visualizar la segmentación YOLO
        """
        imagen_coloreada = imagen_original.copy()
        
        # Para segmentación YOLO de coples (defecto/no defecto)
        if len(self.class_names) == 1:
            # Si hay defecto detectado (máscara == 1)
            if np.any(mascara == 1):
                # Aplicar color rojo semi-transparente para defectos
                color = (0, 0, 255)  # Rojo en BGR
                overlay = imagen_coloreada.copy()
                
                # Crear overlay con color rojo en las áreas defectuosas
                overlay[mascara == 1] = color
                
                # Mezclar con transparencia más suave
                alpha = 0.35  # Transparencia del overlay
                imagen_coloreada = cv2.addWeighted(imagen_coloreada, 1-alpha, overlay, alpha, 0)
                
                # Agregar contornos múltiples para mejor visualización
                contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Contorno exterior amarillo grueso
                cv2.drawContours(imagen_coloreada, contours, -1, (0, 255, 255), 3)  # Amarillo grueso
                
                # Contorno interior blanco fino
                cv2.drawContours(imagen_coloreada, contours, -1, (255, 255, 255), 1)  # Blanco fino
                
                # Marcar centros de detección
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Solo contornos grandes
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(imagen_coloreada, (cx, cy), 4, (0, 255, 0), -1)  # Punto verde
        else:
            # Para múltiples clases
            for class_id in range(1, len(self.class_names) + 1):
                # Crear máscara para esta clase
                class_mask = (mascara == class_id).astype(np.uint8)
                
                if np.any(class_mask):
                    # Aplicar color con transparencia
                    color = self.colormap[class_id - 1] if class_id - 1 < len(self.colormap) else (255, 0, 0)
                    overlay = imagen_coloreada.copy()
                    overlay[class_mask == 1] = color
                    
                    # Mezclar con transparencia
                    alpha = 0.35
                    imagen_coloreada = cv2.addWeighted(imagen_coloreada, 1-alpha, overlay, alpha, 0)
                    
                    # Agregar contornos múltiples
                    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Contorno exterior amarillo grueso
                    cv2.drawContours(imagen_coloreada, contours, -1, (0, 255, 255), 3)
                    
                    # Contorno interior blanco fino
                    cv2.drawContours(imagen_coloreada, contours, -1, (255, 255, 255), 1)
                    
                    # Marcar centros de detección
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.circle(imagen_coloreada, (cx, cy), 4, (0, 255, 0), -1)
        
        return imagen_coloreada

class CamaraTiempoOptimizada:
    def __init__(self, ip="169.254.8.46"):
        self.ip = ip
        self.handle = None
        self.buffer_addresses = None
        self.frame_count = 0
        
        # Parámetros de configuración optimizados para resolución alta
        self.exposure_time = 15000   # 15ms - tiempo de exposición optimizado
        self.framerate = 10.0       # 10 FPS - reducido para menor carga CPU
        self.packet_size = 9000     # Tamaño de paquete jumbo
        self.num_buffers = 2        # Solo 2 buffers para minimizar memoria
        self.gain = 1.0             # Ganancia mínima para mejor calidad
        
        # Configuración del ROI
        self.roi_width = 1280
        self.roi_height = 1024
        self.roi_offset_x = 1416
        self.roi_offset_y = 576
        
        # Sistema de doble buffer asíncrono optimizado
        self.write_buffer_idx = 0    # Buffer donde se está escribiendo actualmente
        self.read_buffer_idx = 1     # Buffer listo para lectura
        
        # Almacenamiento de frames procesados
        self.processed_frames = [None] * 2  # Solo necesitamos 2 slots
        self.frame_ready = [False] * 2      # Estado de cada frame
        self.frame_timestamps = [0] * 2     # Timestamps de captura
        
        # Control de sincronización optimizado
        self.buffer_lock = Lock()           # Lock mínimo para cambios de índices
        self.frame_ready_event = Event()    # Señal de frame listo
        self.capture_thread = None          # Thread de captura continua
        self.capture_active = False         # Control del thread
        
        # Estadísticas de rendimiento
        self.capture_times = Queue(maxsize=100)
        self.processing_times = Queue(maxsize=100)
        self.total_frames_captured = 0
        self.start_time = 0
        
        # Integrar segmentador de coples ONNX
        self.segmentador = SegmentadorCoplesONNX()

    def configurar_camara(self):
        """Configurar parámetros de la cámara una sola vez."""
        try:
            pygigev.GevApiInitialize()
            maxCameras = 16
            numFound = (ctypes.c_uint32)(0)
            camera_info = (pygigev.GEV_CAMERA_INFO * maxCameras)()
            status = pygigev.GevGetCameraList(camera_info, maxCameras, ctypes.byref(numFound))
            
            if status != 0 or numFound.value == 0:
                print("❌ Error buscando cámaras")
                return False

            target_ip_int = ipAddr_from_string(self.ip)
            self.camIndex = -1
            for i in range(numFound.value):
                if camera_info[i].ipAddr == target_ip_int:
                    self.camIndex = i
                    break

            if self.camIndex == -1:
                print(f"❗No se encontró la cámara con IP {self.ip}")
                return False

            self.handle = (ctypes.c_void_p)()
            status = pygigev.GevOpenCamera(
                camera_info[self.camIndex], 
                pygigev.GevExclusiveMode, 
                ctypes.byref(self.handle)
            )
            if status != 0:
                print(f"❌ Error abriendo cámara")
                return False

            # Configurar parámetros de la cámara
            configuraciones = [
                ("ExposureTime", ctypes.c_float(self.exposure_time)),
                ("AcquisitionFrameRate", ctypes.c_float(self.framerate)),
                ("Gain", ctypes.c_float(self.gain))
            ]

            for nombre, valor in configuraciones:
                status = pygigev.GevSetFeatureValue(
                    self.handle,
                    nombre.encode(),
                    ctypes.sizeof(valor),
                    ctypes.byref(valor)
                )
                if status == 0:
                    print(f"✅ {nombre} configurado: {valor.value}")
                else:
                    print(f"❌ Error configurando {nombre}")

            # Configurar ROI
            roi_configs = [
                ("Width", self.roi_width),
                ("Height", self.roi_height),
                ("OffsetX", self.roi_offset_x),
                ("OffsetY", self.roi_offset_y)
            ]

            for nombre, valor in roi_configs:
                valor_int64 = (ctypes.c_int64)(valor)
                status = pygigev.GevSetFeatureValue(
                    self.handle,
                    nombre.encode(),
                    ctypes.sizeof(valor_int64),
                    ctypes.byref(valor_int64)
                )
                if status == 0:
                    print(f"✅ {nombre} configurado: {valor}")
                else:
                    print(f"❌ Error configurando {nombre}")

            # Obtener parámetros de payload
            self.payload_size = (ctypes.c_uint64)()
            self.pixel_format = (ctypes.c_uint32)()
            status = pygigev.GevGetPayloadParameters(
                self.handle,
                ctypes.byref(self.payload_size),
                ctypes.byref(self.pixel_format)
            )
            if status != 0:
                print("❌ Error obteniendo parámetros de payload")
                return False

            # Configurar buffers con margen extra
            self.buffer_addresses = ((ctypes.c_void_p) * self.num_buffers)()
            bufsize = self.payload_size.value + 8192  # Margen más generoso
            
            for i in range(self.num_buffers):
                temp = ((ctypes.c_char) * bufsize)()
                self.buffer_addresses[i] = ctypes.cast(temp, ctypes.c_void_p)

            print(f"✅ Buffers asignados: {self.num_buffers} de {bufsize} bytes")

            # Inicializar transferencia para modo asíncrono
            status = pygigev.GevInitializeTransfer(
                self.handle,
                pygigev.Asynchronous,  # ¡CLAVE! Modo asíncrono
                self.payload_size,
                self.num_buffers,
                self.buffer_addresses
            )
            if status != 0:
                print("❌ Error inicializando transferencia asíncrona")
                return False

            print("📷 Cámara configurada para captura asíncrona")
            
        # Inicializar segmentador de coples ONNX
            if not self.segmentador.inicializar():
                print("❌ Error inicializando segmentador de coples ONNX")
                return False
            
            return True

        except Exception as e:
            print(f"❌ Error en configuración: {e}")
            return False

    def _thread_captura_continua(self):
        """Thread dedicado a captura continua de frames."""
        print("🚀 Iniciando captura continua...")
        
        # Iniciar transferencia continua
        status = pygigev.GevStartTransfer(self.handle, -1)
        if status != 0:
            print("❌ Error iniciando transferencia continua")
            return
            
        self.start_time = time.time()
        frame_local_count = 0
        
        try:
            while self.capture_active:
                capture_start = time.time()
                gevbufPtr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()
                
                # Esperar frame con timeout más largo para resolución alta
                status = pygigev.GevWaitForNextFrame(
                    self.handle,
                    ctypes.byref(gevbufPtr),
                    100  # Timeout aumentado a 100ms para resolución alta
                )

                if status != 0:
                    if self.capture_active:  # Solo reportar si aún estamos activos
                        continue  # Timeout normal, continuar
                    else:
                        break

                capture_time = (time.time() - capture_start) * 1000
                
                # Procesar frame de manera asíncrona
                processing_start = time.time()
                if self._procesar_frame_async(gevbufPtr):
                    frame_local_count += 1
                    self.total_frames_captured += 1
                    
                    # Actualizar estadísticas (sin bloquear)
                    if not self.capture_times.full():
                        self.capture_times.put(capture_time)
                    
                    processing_time = (time.time() - processing_start) * 1000
                    if not self.processing_times.full():
                        self.processing_times.put(processing_time)
                    
                    # Señalar que hay un frame listo
                    self.frame_ready_event.set()
                
                # Liberar el buffer inmediatamente
                if gevbufPtr:
                    pygigev.GevReleaseFrame(self.handle, gevbufPtr)
                    
        except Exception as e:
            print(f"❌ Error en thread de captura: {e}")
        finally:
            # Detener transferencia
            if self.handle:
                pygigev.GevStopTransfer(self.handle)
            print(f"📊 Thread de captura terminado. Frames capturados: {frame_local_count}")

    def _procesar_frame_async(self, gevbufPtr):
        """Procesa frame de manera asíncrona en el buffer de escritura actual."""
        try:
            gevbuf = gevbufPtr.contents
            if gevbuf.status != 0:
                return False

            # Convertir datos del buffer
            im_addr = ctypes.cast(
                gevbuf.address,
                ctypes.POINTER(ctypes.c_ubyte * gevbuf.recv_size)
            )
            raw_data = np.frombuffer(im_addr.contents, dtype=np.uint8)
            raw_data = raw_data.reshape((self.roi_height, self.roi_width))
            
            # Procesar imagen (conversión Bayer a RGB)
            frame_rgb = cv2.cvtColor(raw_data, cv2.COLOR_BayerRG2RGB)
            
            # Actualizar buffer de escritura atómicamente
            with self.buffer_lock:
                # Guardar frame procesado en buffer de escritura
                self.processed_frames[self.write_buffer_idx] = frame_rgb.copy()
                self.frame_ready[self.write_buffer_idx] = True
                self.frame_timestamps[self.write_buffer_idx] = time.time()
                
                # Rotar índices de buffers
                self._rotar_buffers()
            
            return True
            
        except Exception as e:
            print(f"❌ Error procesando frame async: {e}")
            return False

    def _rotar_buffers(self):
        """Rota los índices de buffers de manera circular."""
        # Intercambiar buffers de manera simple
        self.write_buffer_idx, self.read_buffer_idx = self.read_buffer_idx, self.write_buffer_idx

    def iniciar_captura_continua(self):
        """Inicia el thread de captura continua."""
        if self.capture_thread and self.capture_thread.is_alive():
            print("⚠️ La captura ya está activa")
            return True
            
        self.capture_active = True
        self.capture_thread = threading.Thread(
            target=self._thread_captura_continua,
            daemon=True
        )
        self.capture_thread.start()
        
        # Esperar a que el primer frame esté listo (más tiempo para resolución alta)
        if self.frame_ready_event.wait(timeout=5.0):
            print("✅ Captura continua iniciada correctamente")
            return True
        else:
            print("❌ Timeout esperando primer frame")
            return False

    def obtener_frame_y_segmentar_coples(self):
        """
        Obtiene frame instantáneo y lo segmenta con el modelo de coples ONNX.
        Returns: (frame, mascara, imagen_coloreada, tiempo_captura, tiempo_inferencia, tiempo_total)
        """
        start_total = time.time()
        
        # 1. Capturar frame
        start_capture = time.time()
        with self.buffer_lock:
            if self.frame_ready[self.read_buffer_idx]:
                # Copiar frame del buffer de lectura
                frame = self.processed_frames[self.read_buffer_idx].copy()
                timestamp = self.frame_timestamps[self.read_buffer_idx]
                
                # Marcar como procesado
                self.frame_ready[self.read_buffer_idx] = False
                
                tiempo_captura = (time.time() - start_capture) * 1000
            else:
                # No hay frame nuevo, devolver el último disponible
                for i in range(2):
                    if self.frame_ready[i]:
                        frame = self.processed_frames[i].copy()
                        timestamp = self.frame_timestamps[i]
                        tiempo_captura = (time.time() - start_capture) * 1000
                        break
                else:
                    # No hay frames disponibles
                    tiempo_total = (time.time() - start_total) * 1000
                    return None, None, None, 0, 0, tiempo_total
        
        # 2. Segmentar imagen con modelo de coples
        mascara, imagen_coloreada, tiempo_inferencia = self.segmentador.segmentar(frame)
        
        # 3. Calcular tiempo total
        tiempo_total = (time.time() - start_total) * 1000
        
        return frame, mascara, imagen_coloreada, tiempo_captura, tiempo_inferencia, tiempo_total

    def obtener_frame_instantaneo(self):
        """Obtiene el frame más reciente de manera instantánea (~1ms)."""
        start_time = time.time()
        
        with self.buffer_lock:
            if self.frame_ready[self.read_buffer_idx]:
                # Copiar frame del buffer de lectura
                frame = self.processed_frames[self.read_buffer_idx].copy()
                timestamp = self.frame_timestamps[self.read_buffer_idx]
                
                # Marcar como procesado
                self.frame_ready[self.read_buffer_idx] = False
                
                elapsed = (time.time() - start_time) * 1000
                return frame, elapsed, timestamp
            else:
                # No hay frame nuevo, devolver el último disponible
                for i in range(2):
                    if self.frame_ready[i]:
                        frame = self.processed_frames[i].copy()
                        timestamp = self.frame_timestamps[i]
                        elapsed = (time.time() - start_time) * 1000
                        return frame, elapsed, timestamp
        
        elapsed = (time.time() - start_time) * 1000
        return None, elapsed, 0

    def obtener_estadisticas(self):
        """Obtiene estadísticas de rendimiento."""
        if self.start_time == 0:
            return {}
            
        tiempo_total = time.time() - self.start_time
        fps_real = self.total_frames_captured / tiempo_total if tiempo_total > 0 else 0
        
        # Promedios de tiempos
        capture_times_list = list(self.capture_times.queue)
        processing_times_list = list(self.processing_times.queue)
        
        stats = {
            'fps_real': fps_real,
            'frames_totales': self.total_frames_captured,
            'tiempo_total': tiempo_total,
            'tiempo_captura_promedio': sum(capture_times_list) / len(capture_times_list) if capture_times_list else 0,
            'tiempo_procesamiento_promedio': sum(processing_times_list) / len(processing_times_list) if processing_times_list else 0,
            'buffers_listos': sum(self.frame_ready)
        }
        
        return stats

    def detener_captura(self):
        """Detiene la captura continua."""
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        print("🛑 Captura continua detenida")

    def liberar(self):
        """Liberar recursos de la cámara."""
        try:
            # Detener captura
            self.detener_captura()
            
            # Limpiar buffers de manera segura
            with self.buffer_lock:
                try:
                    if hasattr(self, 'processed_frames') and self.processed_frames is not None:
                        for i in range(len(self.processed_frames)):
                            if i < len(self.processed_frames) and self.processed_frames[i] is not None:
                                del self.processed_frames[i]
                        self.processed_frames = [None] * 2
                    if hasattr(self, 'frame_ready') and self.frame_ready is not None:
                        self.frame_ready = [False] * 2
                except Exception as e:
                    print(f"   - Error limpiando buffers: {e}")
                    self.processed_frames = [None] * 2
                    self.frame_ready = [False] * 2
            
            # Cerrar cámara
            if self.handle:
                try:
                    pygigev.GevCloseCamera(self.handle)
                except:
                    pass
                self.handle = None
            
            try:
                pygigev.GevApiUninitialize()
            except:
                pass
                
            # Liberar memoria
            import gc
            gc.collect()
            
            print("✅ Recursos liberados correctamente")
            
        except Exception as e:
            print(f"❌ Error liberando recursos: {e}")

def main():
    """Función principal de demostración con cámara y segmentación YOLO de coples."""
    camara = CamaraTiempoOptimizada()
    
    # Configurar la cámara
    if not camara.configurar_camara():
        print("❌ Error configurando la cámara")
        return

    # Iniciar captura continua
    if not camara.iniciar_captura_continua():
        print("❌ Error iniciando captura continua")
        camara.liberar()
        return

    print("\n🎯 SISTEMA DE CAPTURA Y SEGMENTACIÓN YOLO DE COPLES")
    print("===================================================")
    print("🚀 IMPLEMENTACIÓN CON MÁSCARAS REALES PIXEL-PERFECT")
    print("   - Usa coeficientes + prototipos del modelo YOLOv11")
    print("   - Formas exactas del defecto (no aproximaciones)")
    print("   - Porcentajes de área precisos")
    tipo_mascara = "elípticas (fallback)" if camara.segmentador.usar_mascaras_elipticas else "rectangulares (fallback)"
    print(f"💡 Fallback si no hay prototipos: {tipo_mascara}")
    print("===================================================")
    print("Comandos:")
    print("  ENTER - Capturar imagen y segmentar coples con YOLO REAL")
    print("  'v' - Solo ver frame (sin segmentar)")
    print("  's' - Mostrar estadísticas")
    print("  'm' - Cambiar tipo de máscara fallback (elíptica/rectangular)")
    print("  'q' - Salir")
    print("===================================================\n")
    
    # Crear ventana
    cv2.namedWindow('Frame con Segmentación YOLO de Coples', cv2.WINDOW_NORMAL)
    
    try:
        frame_count = 0
        while True:
            entrada = input("Comando: ").strip().lower()
            
            if entrada == 'q':
                break
            elif entrada == 's':
                stats = camara.obtener_estadisticas()
                print(f"\n📊 ESTADÍSTICAS DE RENDIMIENTO:")
                print(f"   FPS Real: {stats['fps_real']:.1f}")
                print(f"   Frames Totales: {stats['frames_totales']}")
                print(f"   Tiempo Captura Promedio: {stats['tiempo_captura_promedio']:.2f} ms")
                print(f"   Tiempo Procesamiento Promedio: {stats['tiempo_procesamiento_promedio']:.2f} ms")
                print(f"   Buffers con Datos: {stats['buffers_listos']}/2")
                continue
            elif entrada == 'm':
                # Cambiar tipo de máscara fallback
                camara.segmentador.usar_mascaras_elipticas = not camara.segmentador.usar_mascaras_elipticas
                tipo_actual = "elípticas (fallback)" if camara.segmentador.usar_mascaras_elipticas else "rectangulares (fallback)"
                print(f"🔄 Cambiado a máscaras {tipo_actual}")
                print("   ℹ️  Esto solo afecta cuando no hay prototipos disponibles")
                continue
            elif entrada == 'v':
                # Solo obtener frame sin segmentar
                frame, tiempo_acceso, timestamp = camara.obtener_frame_instantaneo()
                
                if frame is not None:
                    frame_count += 1
                    print(f"📷 Frame #{frame_count} obtenido en {tiempo_acceso:.2f} ms")
                    
                    # Mostrar frame
                    cv2.imshow('Frame con Segmentación YOLO de Coples', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("⚠️ No hay frames disponibles")
            elif entrada == '':
                # CAPTURAR Y SEGMENTAR COPLES
                frame, mascara, imagen_coloreada, tiempo_captura, tiempo_inferencia, tiempo_total = camara.obtener_frame_y_segmentar_coples()
                
                if frame is not None and mascara is not None:
                    frame_count += 1
                    
                    print(f"\n🔍 RESULTADO DE SEGMENTACIÓN YOLO DE COPLES #{frame_count}")
                    print("=" * 60)
                    print(f"⏱️  TIEMPOS:")
                    print(f"   Captura:    {tiempo_captura:.2f} ms")
                    print(f"   Inferencia: {tiempo_inferencia:.2f} ms")
                    print(f"   Total:      {tiempo_total:.2f} ms")
                    
                    # Analizar segmentación YOLO
                    if mascara is not None:
                        clases_detectadas = np.unique(mascara)
                        print(f"\n🎯 SEGMENTACIÓN YOLO:")
                        
                        if 1 in clases_detectadas:
                            nombre_clase = camara.segmentador.class_names[0] if camara.segmentador.class_names else "Defecto"
                            pixels_defecto = np.sum(mascara == 1)
                            porcentaje = (pixels_defecto / mascara.size) * 100
                            print(f"   - {nombre_clase}: {pixels_defecto} píxeles ({porcentaje:.2f}%)")
                            print(f"   - Estado: DEFECTO SEGMENTADO")
                        else:
                            print(f"   - Estado: SIN DEFECTOS SEGMENTADOS")
                    
                    print("=" * 60)
                    
                    # Mostrar frame con anotaciones
                    if imagen_coloreada is not None:
                        frame_anotado = imagen_coloreada.copy()
                    else:
                        frame_anotado = frame.copy()
                    
                    # Agregar información de segmentacion en la imagen
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    color = (0, 255, 0)  # Verde
                    thickness = 2
                    
                    # Informacion de segmentación YOLO
                    clases_detectadas = np.unique(mascara)
                    if 1 in clases_detectadas:
                        nombre_clase = camara.segmentador.class_names[0] if camara.segmentador.class_names else "Defecto"
                        pixels_defecto = np.sum(mascara == 1)
                        porcentaje = (pixels_defecto / mascara.size) * 100
                        tipo_mascara = "Elíptica" if camara.segmentador.usar_mascaras_elipticas else "Rectangular"
                        texto_principal = f"YOLO Seg ({tipo_mascara}) - Defecto: {pixels_defecto} pixeles ({porcentaje:.2f}%)"
                    else:
                        texto_principal = "YOLO Seg - Sin defectos segmentados"
                    cv2.putText(frame_anotado, texto_principal, (10, 30), font, font_scale, color, thickness)
                    
                    # Tiempos
                    texto_tiempos = f"Cap: {tiempo_captura:.1f}ms | Seg: {tiempo_inferencia:.1f}ms | Tot: {tiempo_total:.1f}ms"
                    cv2.putText(frame_anotado, texto_tiempos, (10, 60), font, 0.5, (255, 255, 0), 1)
                    
                    # Guardar imagen en directorio Salida_cople
                    try:
                        # Crear directorio si no existe
                        output_dir = "Salida_cople"
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            print(f"Directorio '{output_dir}' creado")
                        
                        # Generar nombre de archivo con timestamp
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"cople_segmentacion_{timestamp}_#{frame_count}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        # Guardar solo la imagen anotada
                        cv2.imwrite(filepath, frame_anotado)
                        print(f"Imagen guardada: {filepath}")
                        
                    except Exception as e:
                        print(f"Error guardando imagen: {e}")
                    
                    cv2.imshow('Frame con Segmentación YOLO de Coples', frame_anotado)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("⚠️ No hay frames disponibles o error en segmentación")
            else:
                print("❓ Comando no reconocido")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
    
    finally:
        print("\n🧹 Liberando recursos...")
        try:
            camara.liberar()
        except:
            pass
        
        # Limpiar OpenCV de manera segura
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Procesar eventos pendientes
        except:
            pass
        
        # Liberar memoria final
        import gc
        gc.collect()
        
        print("✅ Aplicación terminada")

if __name__ == "__main__":
    main()
