"""
Módulo de procesamiento de imágenes para segmentación de coples
Contiene funciones de preprocesamiento y postprocesamiento de imágenes
"""

import cv2
import numpy as np
from config import InferenceConfig, VisualizationConfig


def preprocess_image_segmentation(image, input_size=None):
    """
    Preprocesa la imagen para el modelo de segmentación de coples.
    Optimizado para resolución 1024x1024
    
    Args:
        image (np.ndarray): Imagen original en formato RGB
        input_size (int, optional): Tamaño de entrada del modelo
        
    Returns:
        tuple: (imagen_preprocesada, info_preprocesamiento)
    """
    if input_size is None:
        input_size = InferenceConfig.INPUT_SIZE
    
    try:
        # Redimensionar a resolución del modelo (1024x1024)
        model_size = input_size
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
        model_size = input_size
        fallback = np.zeros((1, 3, model_size, model_size), dtype=np.float32)
        return fallback, (1.0, 0, 0, model_size, model_size)


def postprocess_segmentation(prediction, original_shape, preprocess_info):
    """
    Postprocesa la predicción de segmentación.
    
    Args:
        prediction (np.ndarray): Predicción del modelo
        original_shape (tuple): Forma original de la imagen (H, W, C)
        preprocess_info (tuple): Información del preprocesamiento
        
    Returns:
        np.ndarray: Máscara de segmentación postprocesada
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


def crear_mascara_eliptica(mascara, x_center, y_center, box_w, box_h, img_w, img_h):
    """
    Crea una máscara elíptica ultra-optimizada para velocidad (fallback).
    
    Args:
        mascara (np.ndarray): Máscara donde dibujar
        x_center, y_center (int): Centro de la elipse
        box_w, box_h (int): Dimensiones del bounding box
        img_w, img_h (int): Dimensiones de la imagen
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


def crear_mascara_real(detection, prototipos, cx, cy, w_norm, h_norm, img_w, img_h):
    """
    Crea una máscara REAL usando coeficientes y prototipos del modelo YOLOv11.
    
    Args:
        detection (np.ndarray): Array de detección con coeficientes
        prototipos (np.ndarray): Tensor de prototipos [32, 256, 256]
        cx, cy, w_norm, h_norm (float): Coordenadas normalizadas del bounding box
        img_w, img_h (int): Dimensiones de la imagen original
        
    Returns:
        np.ndarray: Máscara real o None si hay error
    """
    try:
        # Extraer los 32 coeficientes de máscara
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
        
        # Multiplicación matricial: coeficientes @ prototipos
        # coeficientes: [32] @ protos: [32, 256, 256] -> [256, 256]
        mascara_cruda = np.dot(coeficientes, prototipos.reshape(32, -1)).reshape(256, 256)
        
        # Aplicar sigmoid para obtener probabilidades
        mascara_prob = 1 / (1 + np.exp(-mascara_cruda))
        
        # Aplicar umbral para obtener máscara binaria
        mascara_binaria = (mascara_prob > 0.5).astype(np.uint8)
        
        # Recortar máscara según el bounding box de la detección
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


def crear_imagen_coloreada(imagen_original, mascara, class_names):
    """
    Crea una imagen coloreada para visualizar la segmentación YOLO.
    
    Args:
        imagen_original (np.ndarray): Imagen original
        mascara (np.ndarray): Máscara de segmentación
        class_names (list): Lista de nombres de clases
        
    Returns:
        np.ndarray: Imagen coloreada con visualización
    """
    if mascara is None:
        return imagen_original.copy()
    
    imagen_coloreada = imagen_original.copy()
    
    # Para segmentación YOLO de coples (defecto/no defecto)
    if len(class_names) == 1:
        # Si hay defecto detectado (máscara == 1)
        if np.any(mascara == 1):
            # Aplicar color rojo semi-transparente para defectos
            color = VisualizationConfig.DEFECT_COLOR
            overlay = imagen_coloreada.copy()
            
            # Crear overlay con color rojo en las áreas defectuosas
            overlay[mascara == 1] = color
            
            # Mezclar con transparencia
            imagen_coloreada = cv2.addWeighted(
                imagen_coloreada, 1 - VisualizationConfig.OVERLAY_ALPHA, 
                overlay, VisualizationConfig.OVERLAY_ALPHA, 0
            )
            
            # Agregar contornos múltiples para mejor visualización
            contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Contorno exterior amarillo grueso
            cv2.drawContours(imagen_coloreada, contours, -1, 
                           VisualizationConfig.CONTOUR_OUTER_COLOR, 
                           VisualizationConfig.CONTOUR_THICKNESS_OUTER)
            
            # Contorno interior blanco fino
            cv2.drawContours(imagen_coloreada, contours, -1, 
                           VisualizationConfig.CONTOUR_INNER_COLOR, 
                           VisualizationConfig.CONTOUR_THICKNESS_INNER)
            
            # Marcar centros de detección
            for contour in contours:
                if cv2.contourArea(contour) > VisualizationConfig.MIN_CONTOUR_AREA:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(imagen_coloreada, (cx, cy), 
                                 VisualizationConfig.CENTER_POINT_RADIUS, 
                                 VisualizationConfig.CENTER_POINT_COLOR, -1)
    else:
        # Para múltiples clases
        from utils import create_colormap
        colormap = create_colormap(len(class_names))
        
        for class_id in range(1, len(class_names) + 1):
            # Crear máscara para esta clase
            class_mask = (mascara == class_id).astype(np.uint8)
            
            if np.any(class_mask):
                # Aplicar color con transparencia
                color = colormap[class_id - 1] if class_id - 1 < len(colormap) else (255, 0, 0)
                overlay = imagen_coloreada.copy()
                overlay[class_mask == 1] = color
                
                # Mezclar con transparencia
                imagen_coloreada = cv2.addWeighted(
                    imagen_coloreada, 1 - VisualizationConfig.OVERLAY_ALPHA, 
                    overlay, VisualizationConfig.OVERLAY_ALPHA, 0
                )
                
                # Agregar contornos múltiples
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Contorno exterior amarillo grueso
                cv2.drawContours(imagen_coloreada, contours, -1, 
                               VisualizationConfig.CONTOUR_OUTER_COLOR, 
                               VisualizationConfig.CONTOUR_THICKNESS_OUTER)
                
                # Contorno interior blanco fino
                cv2.drawContours(imagen_coloreada, contours, -1, 
                               VisualizationConfig.CONTOUR_INNER_COLOR, 
                               VisualizationConfig.CONTOUR_THICKNESS_INNER)
                
                # Marcar centros de detección
                for contour in contours:
                    if cv2.contourArea(contour) > VisualizationConfig.MIN_CONTOUR_AREA:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(imagen_coloreada, (cx, cy), 
                                     VisualizationConfig.CENTER_POINT_RADIUS, 
                                     VisualizationConfig.CENTER_POINT_COLOR, -1)
    
    return imagen_coloreada


def agregar_anotaciones_imagen(imagen, mascara, class_names, tiempo_captura, tiempo_inferencia, tiempo_total, usar_mascaras_elipticas):
    """
    Agrega anotaciones de texto a la imagen con información de segmentación.
    
    Args:
        imagen (np.ndarray): Imagen base
        mascara (np.ndarray): Máscara de segmentación
        class_names (list): Lista de nombres de clases
        tiempo_captura (float): Tiempo de captura en ms
        tiempo_inferencia (float): Tiempo de inferencia en ms
        tiempo_total (float): Tiempo total en ms
        usar_mascaras_elipticas (bool): Si se usan máscaras elípticas
        
    Returns:
        np.ndarray: Imagen con anotaciones
    """
    frame_anotado = imagen.copy()
    
    # Información de segmentación YOLO
    clases_detectadas = np.unique(mascara)
    if 1 in clases_detectadas:
        nombre_clase = class_names[0] if class_names else "Defecto"
        pixels_defecto = np.sum(mascara == 1)
        porcentaje = (pixels_defecto / mascara.size) * 100
        tipo_mascara = "Elíptica" if usar_mascaras_elipticas else "Rectangular"
        texto_principal = f"YOLO Seg ({tipo_mascara}) - Defecto: {pixels_defecto} pixeles ({porcentaje:.2f}%)"
    else:
        texto_principal = "YOLO Seg - Sin defectos segmentados"
    
    cv2.putText(frame_anotado, texto_principal, (10, 30), 
                VisualizationConfig.FONT, VisualizationConfig.FONT_SCALE, 
                VisualizationConfig.TEXT_COLOR, VisualizationConfig.FONT_THICKNESS)
    
    # Tiempos
    texto_tiempos = f"Cap: {tiempo_captura:.1f}ms | Seg: {tiempo_inferencia:.1f}ms | Tot: {tiempo_total:.1f}ms"
    cv2.putText(frame_anotado, texto_tiempos, (10, 60), 
                VisualizationConfig.FONT, VisualizationConfig.SMALL_FONT_SCALE, 
                VisualizationConfig.TIME_TEXT_COLOR, VisualizationConfig.SMALL_FONT_THICKNESS)
    
    return frame_anotado 