"""
Motor de inferencia ONNX para segmentación de coples
Maneja la carga del modelo, inferencia y procesamiento de máscaras pixel-perfect
"""

import time
import numpy as np
import onnxruntime as ort
import concurrent.futures

# Importar módulos propios
from config import InferenceConfig, GlobalConfig
from utils import (
    get_segmentation_classes, 
    create_colormap, 
    filtrar_detecciones_solapadas,
    calcular_estadisticas_mascara,
    validar_coordenadas,
    limpiar_memoria
)
from image_processor import (
    preprocess_image_segmentation,
    crear_mascara_real,
    crear_mascara_eliptica,
    crear_imagen_coloreada
)


class SegmentadorCoplesONNX:
    """
    Motor de inferencia ONNX para segmentación de coples.
    
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
    
    def __init__(self, model_path=None):
        """
        Inicializa el motor de inferencia.
        
        Args:
            model_path (str, optional): Ruta del modelo ONNX. Si no se proporciona, usa la configuración por defecto.
        """
        self.model_path = model_path or InferenceConfig.DEFAULT_MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_name = None
        self.class_names = None
        self.colormap = None
        self.inicializado = False
        
        # Control de procesamiento
        self.procesamiento_activo = GlobalConfig.PROCESSING_ACTIVE
        self.usar_mascaras_elipticas = GlobalConfig.USE_ELLIPTICAL_MASKS
        
    def inicializar(self):
        """
        Inicializa el modelo ONNX y las clases de segmentación de coples.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Verificar que el modelo existe
            GlobalConfig.get_model_path()  # Esto lanza excepción si no existe
            
            # Cargar modelo ONNX con optimizaciones
            print("🧠 Cargando modelo de segmentación de coples ONNX...")
            
            # Configurar opciones de ONNX para mejor rendimiento
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = InferenceConfig.INTRA_OP_THREADS
            session_options.inter_op_num_threads = InferenceConfig.INTER_OP_THREADS
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=InferenceConfig.PROVIDERS
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
        Procesa una imagen RGB con el modelo de segmentación YOLO.
        
        Implementa segmentación REAL con:
        - Máscaras pixel-perfect usando coeficientes y prototipos del modelo
        - Decodificación de máscaras YOLOv11 real
        - Contornos múltiples (amarillo + blanco)
        - Puntos centrales marcados
        - Estadísticas detalladas de máscaras reales
        
        Args:
            image_rgb (np.ndarray): Imagen RGB de entrada
            
        Returns:
            tuple: (mascara_segmentacion, imagen_coloreada, tiempo_inferencia_ms)
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
            
            # Preprocesar imagen para el modelo
            input_tensor, preprocess_info = preprocess_image_segmentation(
                image_rgb, 
                input_size=InferenceConfig.INPUT_SIZE
            )
            
            # Realizar inferencia con timeout seguro
            results = self._ejecutar_inferencia_segura(input_tensor)
            
            if results is None:
                # Error en inferencia, retornar sin defectos
                h, w = image_rgb.shape[:2]
                mascara = np.zeros((h, w), dtype=np.uint8)
                imagen_coloreada = image_rgb.copy()
                tiempo_inferencia = (time.time() - start_inference) * 1000
                return mascara, imagen_coloreada, tiempo_inferencia
            
            # Verificar si tenemos ambas salidas del modelo
            if len(results) >= 2:
                prediction = results[0]  # Detecciones + coeficientes: [1, 37, 21504]
                prototipos = results[1]   # Prototipos de máscara: [1, 32, 256, 256]
                print(f"✅ Modelo con máscaras reales: detecciones {prediction.shape}, prototipos {prototipos.shape}")
            else:
                prediction = results[0]  # Solo detecciones
                prototipos = None
                print(f"⚠️ Modelo sin prototipos, usando detecciones: {prediction.shape}")
            
            # Procesar salida YOLO SEGMENTACIÓN
            mascara = self._procesar_salida_yolo(prediction, prototipos, image_rgb.shape)
            
            # Crear imagen coloreada para visualización
            imagen_coloreada = crear_imagen_coloreada(image_rgb, mascara, self.class_names)
            
            # Liberar memoria explícitamente
            self._limpiar_memoria_inferencia(input_tensor, results)
            
            tiempo_inferencia = (time.time() - start_inference) * 1000
            return mascara, imagen_coloreada, tiempo_inferencia
            
        except Exception as e:
            print(f"❌ Error en detección de coples: {e}")
            return None, None, 0
    
    def _ejecutar_inferencia_segura(self, input_tensor):
        """
        Ejecuta la inferencia con timeout y manejo de errores.
        
        Args:
            input_tensor (np.ndarray): Tensor de entrada
            
        Returns:
            list: Resultados de la inferencia o None si hay error
        """
        try:
            # Configurar ONNX para mejor rendimiento
            ort.set_default_logger_severity(3)  # Reducir logs
            
            def run_inference():
                return self.session.run(None, {self.input_name: input_tensor})
            
            # Ejecutar inferencia con timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_inference)
                try:
                    results = future.result(timeout=InferenceConfig.INFERENCE_TIMEOUT)
                    return results
                except concurrent.futures.TimeoutError:
                    print("⚠️ Timeout en inferencia, asumiendo sin defectos")
                    future.cancel()
                    return None
                
        except Exception as e:
            print(f"⚠️ Error en inferencia: {e}")
            return None
    
    def _procesar_salida_yolo(self, prediction, prototipos, image_shape):
        """
        Procesa la salida del modelo YOLO para generar máscaras.
        
        Args:
            prediction (np.ndarray): Predicciones del modelo
            prototipos (np.ndarray): Prototipos de máscara (puede ser None)
            image_shape (tuple): Forma de la imagen original
            
        Returns:
            np.ndarray: Máscara de segmentación procesada
        """
        h, w = image_shape[:2]
        mascara = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Transponer para facilitar procesamiento: [1, 21504, 37]
            predictions = prediction.transpose(0, 2, 1)
            predictions = predictions[0]  # [21504, 37]
            
            # Filtrar detecciones con confianza > umbral
            conf_mask = predictions[:, 4] > InferenceConfig.CONFIDENCE_THRESHOLD
            valid_predictions = predictions[conf_mask]
            
            # Limitar número de detecciones
            if len(valid_predictions) > InferenceConfig.MAX_DETECTIONS:
                indices = np.argsort(valid_predictions[:, 4])[-InferenceConfig.MAX_DETECTIONS:]
                valid_predictions = valid_predictions[indices]
            
            # Filtrar detecciones solapadas
            detecciones_originales = len(valid_predictions)
            if len(valid_predictions) > 1:
                valid_predictions = filtrar_detecciones_solapadas(valid_predictions)
            
            if len(valid_predictions) > 0:
                if detecciones_originales > len(valid_predictions):
                    print(f"🎯 Detecciones de segmentación encontradas: {len(valid_predictions)} (filtradas {detecciones_originales - len(valid_predictions)} solapadas)")
                else:
                    print(f"🎯 Detecciones de segmentación encontradas: {len(valid_predictions)}")
                
                # Crear máscaras para cada detección
                mascara = self._crear_mascaras_detecciones(valid_predictions, prototipos, w, h)
                
                # Mostrar estadísticas finales
                self._mostrar_estadisticas_mascara(mascara, prototipos is not None)
            else:
                print("✅ No se encontraron defectos en segmentación")
                
        except Exception as e:
            print(f"⚠️ Error procesando segmentación YOLO: {e}")
            print(f"   Forma de predicción: {prediction.shape}")
            mascara = np.zeros((h, w), dtype=np.uint8)
        
        return mascara
    
    def _crear_mascaras_detecciones(self, detecciones, prototipos, w, h):
        """
        Crea máscaras para las detecciones válidas.
        
        Args:
            detecciones (np.ndarray): Detecciones válidas
            prototipos (np.ndarray): Prototipos de máscara (puede ser None)
            w, h (int): Dimensiones de la imagen
            
        Returns:
            np.ndarray: Máscara combinada de todas las detecciones
        """
        mascara = np.zeros((h, w), dtype=np.uint8)
        start_mask_time = time.time()
        
        for i, detection in enumerate(detecciones):
            # Timeout para procesamiento de máscaras
            if time.time() - start_mask_time > InferenceConfig.MASK_PROCESSING_TIMEOUT:
                print(f"   ⚠️ Timeout procesando máscaras, procesadas {i}/{len(detecciones)}")
                break
            
            # Procesar detección individual
            mascara_individual = self._procesar_deteccion_individual(
                detection, prototipos, w, h
            )
            
            if mascara_individual is not None:
                mascara = np.logical_or(mascara, mascara_individual).astype(np.uint8)
        
        return mascara
    
    def _procesar_deteccion_individual(self, detection, prototipos, w, h):
        """
        Procesa una detección individual para crear su máscara.
        
        Args:
            detection (np.ndarray): Detección individual
            prototipos (np.ndarray): Prototipos de máscara (puede ser None)
            w, h (int): Dimensiones de la imagen
            
        Returns:
            np.ndarray: Máscara de la detección o None si hay error
        """
        try:
            # Extraer coordenadas
            cx, cy, w_norm, h_norm = detection[:4]
            conf = detection[4]
            
            # Validar y normalizar coordenadas
            cx, cy, w_norm, h_norm, x1, y1, x2, y2 = validar_coordenadas(
                cx, cy, w_norm, h_norm, w, h
            )
            
            # Verificar que las coordenadas sean válidas
            if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
                print(f"   - Coordenadas fuera de rango: bbox=[{x1},{y1},{x2},{y2}] (imagen: {w}x{h})")
                return None
            
            x_center = int(cx * w)
            y_center = int(cy * h)
            box_w = int(w_norm * w)
            box_h = int(h_norm * h)
            
            pixels_antes = 0  # Para estadísticas
            
            # Crear máscara según el tipo disponible
            if prototipos is not None:
                # Crear máscara REAL usando coeficientes y prototipos
                mascara_real = crear_mascara_real(
                    detection, prototipos[0], cx, cy, w_norm, h_norm, w, h
                )
                if mascara_real is not None:
                    tipo_mascara = "REAL (coeficientes+prototipos)"
                    return mascara_real
                else:
                    # Fallback a máscara elíptica si falla el procesamiento real
                    mascara_fallback = np.zeros((h, w), dtype=np.uint8)
                    if self.usar_mascaras_elipticas:
                        crear_mascara_eliptica(mascara_fallback, x_center, y_center, box_w, box_h, w, h)
                        tipo_mascara = "elíptica (fallback)"
                    else:
                        mascara_fallback[y1:y2, x1:x2] = 1
                        tipo_mascara = "rectangular (fallback)"
                    return mascara_fallback
            else:
                # Sin prototipos, usar método aproximado
                mascara_aproximada = np.zeros((h, w), dtype=np.uint8)
                if self.usar_mascaras_elipticas and len(detection) <= 2:
                    crear_mascara_eliptica(mascara_aproximada, x_center, y_center, box_w, box_h, w, h)
                    tipo_mascara = "elíptica"
                else:
                    mascara_aproximada[y1:y2, x1:x2] = 1
                    tipo_mascara = "rectangular" if not self.usar_mascaras_elipticas else "rectangular (múltiples)"
                
                pixels_agregados = np.sum(mascara_aproximada)
                print(f"   - Máscara {tipo_mascara} creada: conf={conf:.3f}, centro=[{x_center},{y_center}], tamaño=[{box_w}x{box_h}], píxeles={pixels_agregados}")
                
                return mascara_aproximada
                
        except Exception as e:
            print(f"   ❌ Error procesando detección individual: {e}")
            return None
    
    def _mostrar_estadisticas_mascara(self, mascara, tiene_prototipos):
        """
        Muestra estadísticas detalladas de la máscara final.
        
        Args:
            mascara (np.ndarray): Máscara final
            tiene_prototipos (bool): Si el modelo tiene prototipos
        """
        if np.any(mascara):
            stats = calcular_estadisticas_mascara(mascara)
            
            mascara_tipo = "REAL" if tiene_prototipos else "Aproximada"
            print(f"📊 Estadísticas de máscara {mascara_tipo}:")
            print(f"   - Píxeles defectuosos: {stats['defect_pixels']} ({stats['percentage']:.2f}%)")
            print(f"   - Regiones detectadas: {stats['num_regions']}")
            print(f"   - Área promedio: {stats['avg_area']:.1f} píxeles")
            print(f"   - Área máxima: {stats['max_area']:.1f} píxeles")
    
    def _limpiar_memoria_inferencia(self, input_tensor, results):
        """
        Limpia la memoria utilizada en la inferencia.
        
        Args:
            input_tensor (np.ndarray): Tensor de entrada
            results (list): Resultados de la inferencia
        """
        try:
            del input_tensor
            if results:
                for result in results:
                    if result is not None:
                        del result
            limpiar_memoria()
        except:
            pass
    
    def cambiar_tipo_mascara(self):
        """
        Cambia el tipo de máscara fallback (elíptica/rectangular).
        
        Returns:
            str: Tipo de máscara actual
        """
        self.usar_mascaras_elipticas = not self.usar_mascaras_elipticas
        tipo_actual = "elípticas (fallback)" if self.usar_mascaras_elipticas else "rectangulares (fallback)"
        print(f"🔄 Cambiado a máscaras {tipo_actual}")
        print("   ℹ️  Esto solo afecta cuando no hay prototipos disponibles")
        return tipo_actual
    
    def activar_procesamiento(self, activo=True):
        """
        Activa o desactiva el procesamiento de inferencia.
        
        Args:
            activo (bool): True para activar, False para desactivar
        """
        self.procesamiento_activo = activo
        estado = "activado" if activo else "desactivado"
        print(f"🔄 Procesamiento de inferencia {estado}")
    
    def obtener_info_modelo(self):
        """
        Obtiene información detallada del modelo cargado.
        
        Returns:
            dict: Información del modelo
        """
        if not self.inicializado:
            return {}
        
        info = {
            'ruta_modelo': self.model_path,
            'clases': self.class_names,
            'num_clases': len(self.class_names) if self.class_names else 0,
            'nombre_entrada': self.input_name,
            'forma_entrada': self.session.get_inputs()[0].shape,
            'nombres_salida': [output.name for output in self.session.get_outputs()],
            'formas_salida': [output.shape for output in self.session.get_outputs()],
            'procesamiento_activo': self.procesamiento_activo,
            'usar_mascaras_elipticas': self.usar_mascaras_elipticas,
            'umbral_confianza': InferenceConfig.CONFIDENCE_THRESHOLD,
            'max_detecciones': InferenceConfig.MAX_DETECTIONS
        }
        
        return info
    
    def mostrar_configuracion(self):
        """Muestra la configuración actual del motor de inferencia."""
        info = self.obtener_info_modelo()
        
        print(f"\n🧠 CONFIGURACIÓN DEL MOTOR DE INFERENCIA:")
        print(f"   Modelo: {info.get('ruta_modelo', 'N/A')}")
        print(f"   Clases: {info.get('num_clases', 0)} - {info.get('clases', [])}")
        print(f"   Entrada: {info.get('nombre_entrada', 'N/A')} {info.get('forma_entrada', 'N/A')}")
        print(f"   Salidas: {len(info.get('nombres_salida', []))}")
        print(f"   Umbral confianza: {info.get('umbral_confianza', 'N/A')}")
        print(f"   Máx. detecciones: {info.get('max_detecciones', 'N/A')}")
        print(f"   Procesamiento: {'ACTIVO' if info.get('procesamiento_activo', False) else 'DESACTIVADO'}")
        print(f"   Máscaras fallback: {'Elípticas' if info.get('usar_mascaras_elipticas', False) else 'Rectangulares'}")
        
        if self.inicializado:
            print("   Estado: INICIALIZADO")
        else:
            print("   Estado: NO INICIALIZADO") 