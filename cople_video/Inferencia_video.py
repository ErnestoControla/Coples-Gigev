"""
Sistema de Inferencia de Video para Segmentación de Coples
Basado en el sistema existente pero adaptado para procesamiento de video en tiempo real
"""

import cv2
import time
import numpy as np
import onnxruntime as ort
import os
import json
import threading
from threading import Event, Lock
from queue import Queue
import sys
import ctypes
from datetime import datetime

# Importar el código GigE común
sys.path.append(os.path.dirname(__file__) + "/../../gigev_common")
import pygigev

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
        model_size = 1024
        image_resized = cv2.resize(image, (model_size, model_size), interpolation=cv2.INTER_AREA)
        
        # Normalizar pixel values to [0,1]
        new_image = image_resized.astype(np.float32, copy=False) / 255.0
        
        # Cambiar dimensiones de HWC a CHW
        new_image = np.transpose(new_image, (2, 0, 1))
        
        # Agregar batch dimension
        new_image = np.expand_dims(new_image, axis=0)
        
        return new_image, (1.0, 0, 0, model_size, model_size)
        
    except Exception as e:
        print(f"⚠️ Error en preprocesamiento: {e}")
        model_size = 1024
        fallback = np.zeros((1, 3, model_size, model_size), dtype=np.float32)
        return fallback, (1.0, 0, 0, model_size, model_size)

def get_segmentation_classes():
    """Obtiene las clases de segmentación desde el archivo local"""
    classes_file = "/home/ernesto/Documentos/Proyectos/Coples/coples_seg_clases.txt"
    
    try:
        if not os.path.exists(classes_file):
            print(f"❌ Error: No se encontró el archivo de clases: {classes_file}")
            return ["Defecto"]  # Clase por defecto
        
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"✅ Cargadas {len(classes)} clases de segmentación")
        return classes
        
    except Exception as e:
        print(f"❌ Error leyendo clases: {e}")
        return ["Defecto"]  # Clase por defecto

class SegmentadorVideoCoples:
    """Segmentador optimizado para video en tiempo real"""
    
    def __init__(self, model_path="/home/ernesto/Documentos/Proyectos/Coples/coples_seg1C8V.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.class_names = None
        self.inicializado = False
        
    def inicializar(self):
        """Inicializa el modelo ONNX"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Error: No se encontró el modelo: {self.model_path}")
                return False
            
            print("🧠 Cargando modelo de segmentación de coples para video...")
            
            # Configurar ONNX para rendimiento de video
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 2
            session_options.inter_op_num_threads = 2
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=['CPUExecutionProvider']
            )
            
            # Obtener información del modelo
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            
            print(f"✅ Modelo cargado - Entrada: {input_shape}")
            
            # Cargar clases
            self.class_names = get_segmentation_classes()
            
            self.inicializado = True
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando segmentador: {e}")
            return False
    
    def segmentar_frame(self, image_rgb):
        """
        Segmenta un frame de video
        Returns: (mascara, num_defectos, tiene_defectos, tiempo_inferencia_ms)
        """
        if not self.inicializado:
            return None, 0, False, 0
            
        try:
            start_inference = time.time()
            
            # Preprocesar imagen
            input_tensor, preprocess_info = preprocess_image_segmentation(image_rgb, input_size=1024)
            
            # Ejecutar inferencia con timeout
            try:
                import concurrent.futures
                
                def run_inference():
                    return self.session.run(None, {self.input_name: input_tensor})
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_inference)
                    results = future.result(timeout=1.5)  # Timeout más corto para video
                    
                    if len(results) >= 2:
                        prediction = results[0]
                        prototipos = results[1]
                    else:
                        prediction = results[0]
                        prototipos = None
                        
            except concurrent.futures.TimeoutError:
                print("⚠️ Timeout en inferencia de video")
                h, w = image_rgb.shape[:2]
                mascara = np.zeros((h, w), dtype=np.uint8)
                tiempo_inferencia = (time.time() - start_inference) * 1000
                return mascara, 0, False, tiempo_inferencia
            
            # Procesar resultados
            mascara = self._procesar_prediccion_video(prediction, prototipos, image_rgb.shape)
            
            # Contar defectos (regiones conectadas)
            if np.any(mascara):
                contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Filtrar contornos pequeños (ruido)
                contours_validos = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
                num_defectos = len(contours_validos)
                tiene_defectos = num_defectos > 0
            else:
                num_defectos = 0
                tiene_defectos = False
            
            tiempo_inferencia = (time.time() - start_inference) * 1000
            return mascara, num_defectos, tiene_defectos, tiempo_inferencia
            
        except Exception as e:
            print(f"❌ Error en segmentación de video: {e}")
            h, w = image_rgb.shape[:2]
            mascara = np.zeros((h, w), dtype=np.uint8)
            return mascara, 0, False, 0
    
    def _procesar_prediccion_video(self, prediction, prototipos, image_shape):
        """Procesa la predicción del modelo para video (versión optimizada)"""
        h, w = image_shape[:2]
        mascara = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Transponer para facilitar procesamiento
            predictions = prediction.transpose(0, 2, 1)
            predictions = predictions[0]
            
            # Filtrar detecciones con confianza > 0.4 (más restrictivo para video)
            conf_mask = predictions[:, 4] > 0.4
            valid_predictions = predictions[conf_mask]
            
            # Limitar detecciones para rendimiento de video
            if len(valid_predictions) > 5:
                indices = np.argsort(valid_predictions[:, 4])[-5:]
                valid_predictions = valid_predictions[indices]
            
            for detection in valid_predictions:
                cx, cy, w_norm, h_norm = detection[:4]
                
                # Normalizar coordenadas si es necesario
                if cx > 1:
                    cx = cx / 1024.0
                if cy > 1:
                    cy = cy / 1024.0
                if w_norm > 1:
                    w_norm = w_norm / 1024.0
                if h_norm > 1:
                    h_norm = h_norm / 1024.0
                
                # Convertir a píxeles
                x_center = int(cx * w)
                y_center = int(cy * h)
                box_w = int(w_norm * w)
                box_h = int(h_norm * h)
                
                # Crear máscara elíptica simple (más rápido para video)
                x1 = max(0, x_center - box_w // 2)
                y1 = max(0, y_center - box_h // 2)
                x2 = min(w, x_center + box_w // 2)
                y2 = min(h, y_center + box_h // 2)
                
                if x2 > x1 and y2 > y1:
                    # Usar elipse para mejor apariencia
                    a = max(1, box_w // 2 - 2)
                    b = max(1, box_h // 2 - 2)
                    cv2.ellipse(mascara, (x_center, y_center), (a, b), 0, 0, 360, 1, -1)
                    
        except Exception as e:
            print(f"⚠️ Error procesando predicción: {e}")
            
        return mascara

class CamaraVideoCoples:
    """Controlador de cámara optimizado para video"""
    
    def __init__(self, ip="169.254.8.46"):
        self.ip = ip
        self.handle = None
        self.buffer_addresses = None
        self.capture_active = False
        self.capture_thread = None
        
        # Parámetros optimizados para video
        self.exposure_time = 12000  # Más rápido para video fluido
        self.framerate = 15.0       # FPS para video
        self.packet_size = 9000
        self.num_buffers = 3        # Más buffers para video
        self.gain = 1.5
        
        # ROI igual que el sistema original
        self.roi_width = 1280
        self.roi_height = 1024
        self.roi_offset_x = 1416
        self.roi_offset_y = 576
        
        # Buffer para video
        self.frame_lock = Lock()
        self.current_frame = None
        self.frame_ready = Event()
        
    def configurar_camara(self):
        """Configura la cámara para video"""
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

            # Configurar parámetros
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

            # Configurar buffers
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

            self.buffer_addresses = ((ctypes.c_void_p) * self.num_buffers)()
            bufsize = self.payload_size.value + 8192
            
            for i in range(self.num_buffers):
                temp = ((ctypes.c_char) * bufsize)()
                self.buffer_addresses[i] = ctypes.cast(temp, ctypes.c_void_p)

            print(f"✅ Buffers configurados: {self.num_buffers} de {bufsize} bytes")

            # Inicializar transferencia
            status = pygigev.GevInitializeTransfer(
                self.handle,
                pygigev.Asynchronous,
                self.payload_size,
                self.num_buffers,
                self.buffer_addresses
            )
            if status != 0:
                print("❌ Error inicializando transferencia")
                return False

            print("📷 Cámara configurada para video")
            return True

        except Exception as e:
            print(f"❌ Error en configuración: {e}")
            return False
    
    def iniciar_captura_video(self):
        """Inicia la captura continua para video"""
        if self.capture_thread and self.capture_thread.is_alive():
            return True
            
        self.capture_active = True
        self.capture_thread = threading.Thread(
            target=self._thread_captura_video,
            daemon=True
        )
        self.capture_thread.start()
        
        # Esperar primer frame
        if self.frame_ready.wait(timeout=3.0):
            print("✅ Captura de video iniciada")
            return True
        else:
            print("❌ Timeout esperando primer frame de video")
            return False
    
    def _thread_captura_video(self):
        """Thread de captura optimizado para video"""
        print("🚀 Iniciando captura de video...")
        
        status = pygigev.GevStartTransfer(self.handle, -1)
        if status != 0:
            print("❌ Error iniciando transferencia de video")
            return
            
        try:
            while self.capture_active:
                gevbufPtr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()
                
                status = pygigev.GevWaitForNextFrame(
                    self.handle,
                    ctypes.byref(gevbufPtr),
                    50  # Timeout más corto para video fluido
                )

                if status != 0:
                    if self.capture_active:
                        continue
                    else:
                        break

                gevbuf = gevbufPtr.contents
                if gevbuf.status == 0:
                    # Convertir frame
                    im_addr = ctypes.cast(
                        gevbuf.address,
                        ctypes.POINTER(ctypes.c_ubyte * gevbuf.recv_size)
                    )
                    raw_data = np.frombuffer(im_addr.contents, dtype=np.uint8)
                    raw_data = raw_data.reshape((self.roi_height, self.roi_width))
                    
                    # Convertir a RGB
                    frame_rgb = cv2.cvtColor(raw_data, cv2.COLOR_BayerRG2RGB)
                    
                    # Actualizar frame actual
                    with self.frame_lock:
                        self.current_frame = frame_rgb.copy()
                        self.frame_ready.set()
                
                # Liberar buffer
                if gevbufPtr:
                    pygigev.GevReleaseFrame(self.handle, gevbufPtr)
                    
        except Exception as e:
            print(f"❌ Error en captura de video: {e}")
        finally:
            if self.handle:
                pygigev.GevStopTransfer(self.handle)
            print("📊 Captura de video terminada")
    
    def obtener_frame(self):
        """Obtiene el frame más reciente"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def detener_captura(self):
        """Detiene la captura de video"""
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
    
    def liberar(self):
        """Libera recursos"""
        try:
            self.detener_captura()
            
            if self.handle:
                pygigev.GevCloseCamera(self.handle)
                self.handle = None
            
            pygigev.GevApiUninitialize()
            print("✅ Recursos de cámara liberados")
            
        except Exception as e:
            print(f"❌ Error liberando recursos: {e}")

def agregar_anotaciones_video(frame, mascara, num_defectos, tiene_defectos, tiempo_inferencia, fps):
    """Agrega anotaciones específicas para video"""
    frame_anotado = frame.copy()
    
    # Configuración de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_grande = 1.2
    font_scale_normal = 0.8
    font_scale_pequeno = 0.6
    thickness_grande = 3
    thickness_normal = 2
    
    # Determinar estado y color
    if tiene_defectos:
        estado = "NG"
        color_estado = (0, 0, 255)  # Rojo
        # Dibujar contornos de defectos
        contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                cv2.drawContours(frame_anotado, [contour], -1, (0, 255, 255), 2)
                # Overlay semi-transparente
                overlay = frame_anotado.copy()
                cv2.fillPoly(overlay, [contour], (0, 0, 255))
                frame_anotado = cv2.addWeighted(frame_anotado, 0.8, overlay, 0.2, 0)
    else:
        estado = "OK"
        color_estado = (0, 255, 0)  # Verde
    
    # Estado principal (esquina superior izquierda)
    cv2.putText(frame_anotado, estado, (30, 80), font, font_scale_grande, 
                color_estado, thickness_grande)
    
    # Número de defectos (debajo del estado)
    texto_defectos = f"Defectos: {num_defectos}"
    cv2.putText(frame_anotado, texto_defectos, (30, 130), font, font_scale_normal, 
                (255, 255, 255), thickness_normal)
    
    # Información técnica (esquina superior derecha)
    h, w = frame.shape[:2]
    texto_fps = f"FPS: {fps:.1f}"
    texto_inference = f"Inf: {tiempo_inferencia:.1f}ms"
    
    cv2.putText(frame_anotado, texto_fps, (w-200, 30), font, font_scale_pequeno, 
                (255, 255, 0), 1)
    cv2.putText(frame_anotado, texto_inference, (w-200, 60), font, font_scale_pequeno, 
                (255, 255, 0), 1)
    
    # Timestamp (esquina inferior derecha)
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame_anotado, timestamp, (w-150, h-20), font, font_scale_pequeno, 
                (255, 255, 255), 1)
    
    return frame_anotado

def verificar_video_valido(filepath):
    """Verifica si el video generado es válido y reproducible"""
    try:
        # Intentar abrir el video con OpenCV
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return False, "No se pudo abrir el video"
        
        # Verificar propiedades básicas
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Intentar leer un frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, "No se pudo leer ningún frame"
        
        if frame_count == 0:
            return False, "Video sin frames"
            
        print(f"   ✅ Video válido: {frame_count} frames, {fps} FPS, {width}x{height}")
        return True, "Video válido"
        
    except Exception as e:
        return False, f"Error verificando video: {e}"

def iniciar_grabacion_video(output_dir, fps_grabacion=5.0, codec_preferido='MJPG'):
    """Inicia la grabación de video con timestamp"""
    try:
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inferencia_cople_{timestamp}.avi"  # Cambiar a AVI para mejor compatibilidad
        filepath = os.path.join(output_dir, filename)
        
        # Configurar codec y VideoWriter
        # Ordenar codecs poniendo el preferido primero
        all_codecs = [
            ('MJPG', 'MJPG'),  # Motion JPEG - MUY compatible
            ('XVID', 'XVID'),  # AVI con XVID 
            ('H264', 'H264'),  # H.264 si está disponible
            ('MP4V', 'MP4V'),  # MPEG-4 fallback
        ]
        
        # Reorganizar para poner el codec preferido primero
        codecs_to_try = []
        for codec_name, fourcc_str in all_codecs:
            if codec_name == codec_preferido:
                codecs_to_try.insert(0, (codec_name, fourcc_str))
            else:
                codecs_to_try.append((codec_name, fourcc_str))
        
        # Dimensiones del video (mismo tamaño que los frames)
        frame_width = 1280
        frame_height = 1024
        
        video_writer = None
        codec_used = None
        
        for codec_name, fourcc_str in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                video_writer = cv2.VideoWriter(filepath, fourcc, fps_grabacion, (frame_width, frame_height))
                
                if video_writer.isOpened():
                    codec_used = codec_name
                    print(f"🎬 Grabación iniciada: {filename}")
                    print(f"   📊 FPS: {fps_grabacion}, Resolución: {frame_width}x{frame_height}")
                    print(f"   🎞️  Codec: {codec_used}")
                    return video_writer, filename
                else:
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                        
            except Exception as e:
                print(f"   ⚠️ Codec {codec_name} no disponible: {e}")
                if video_writer:
                    video_writer.release()
                    video_writer = None
                continue
        
        print("❌ Error: No se pudo inicializar VideoWriter con ningún codec")
        return None, None
            
    except Exception as e:
        print(f"❌ Error iniciando grabación: {e}")
        return None, None

def detener_grabacion_video(video_writer, filename):
    """Detiene la grabación de video y libera recursos de forma segura"""
    try:
        if video_writer is not None and video_writer.isOpened():
            # Asegurarse de que todos los frames se escriban
            video_writer.release()
            # Dar tiempo suficiente para que se complete la escritura
            time.sleep(0.5)
            
            # Verificar que el archivo existe y tiene contenido
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                if file_size > 1024:  # Al menos 1KB
                    print(f"🎬 Grabación finalizada: {filename} ({file_size/1024:.1f} KB)")
                    
                    # Verificar si el video es reproducible
                    es_valido, mensaje = verificar_video_valido(filename)
                    if not es_valido:
                        print(f"   ⚠️ {mensaje}")
                        return False
                    
                    return True
                else:
                    print(f"⚠️ Video creado pero muy pequeño: {filename} ({file_size} bytes)")
                    return False
            else:
                print(f"❌ Archivo de video no encontrado: {filename}")
                return False
        elif video_writer is not None:
            print(f"⚠️ VideoWriter ya estaba cerrado: {filename}")
            return True
    except Exception as e:
        print(f"❌ Error deteniendo grabación: {e}")
        try:
            # Intentar liberar de cualquier forma
            if video_writer is not None:
                video_writer.release()
            time.sleep(0.2)
        except:
            pass
    return False

def main():
    """Función principal del sistema de video"""
    print("🎬 SISTEMA DE INFERENCIA DE VIDEO PARA COPLES")
    print("=" * 50)
    print("Basado en modelo coples_seg1C8V.onnx")
    print("Mostrando OK/NG en tiempo real")
    print("=" * 50)
    
    # Crear directorio de salida con ruta absoluta
    output_dir = "/home/ernesto/Documentos/Proyectos/Coples/salida_video"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Directorio '{output_dir}' creado")
    
    # Inicializar componentes
    camara = CamaraVideoCoples()
    segmentador = SegmentadorVideoCoples()
    
    # Configurar cámara
    if not camara.configurar_camara():
        print("❌ Error configurando cámara")
        return
    
    # Inicializar segmentador
    if not segmentador.inicializar():
        print("❌ Error inicializando segmentador")
        camara.liberar()
        return
    
    # Iniciar captura
    if not camara.iniciar_captura_video():
        print("❌ Error iniciando captura de video")
        camara.liberar()
        return
    
    print("\n🎥 INICIANDO VIDEO EN TIEMPO REAL")
    print("Comandos:")
    print("  's' - Guardar frame actual")
    print("  'r' - Iniciar/Detener grabación de video")
    print("  'c' - Cambiar codec (MJPG/XVID)")
    print("  'q' - Salir")
    print("-" * 30)
    print("💡 Sugerencia: Si el video no es reproducible, prueba 'c' para cambiar codec")
    
    # Crear ventana
    cv2.namedWindow('Video Inferencia Coples', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Inferencia Coples', 1000, 800)
    
    # Variables de control
    frame_count = 0
    fps_counter = 0
    fps_timer = time.time()
    fps_actual = 0
    
    # Variables de grabación de video
    video_writer = None
    grabando_video = False
    video_filename = None
    fps_grabacion = 5.0  # FPS para grabación (ajustado según rendimiento real)
    tiempo_ultimo_frame_grabado = 0
    intervalo_frame = 1.0 / fps_grabacion  # Intervalo entre frames para grabación
    codec_preferido = 'MJPG'  # Codec por defecto (más compatible)
    
    try:
        while True:
            # Obtener frame
            frame = camara.obtener_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            fps_counter += 1
            
            # Calcular FPS cada segundo
            if time.time() - fps_timer >= 1.0:
                fps_actual = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Segmentar frame
            mascara, num_defectos, tiene_defectos, tiempo_inferencia = segmentador.segmentar_frame(frame)
            
            # Crear frame anotado
            if mascara is not None:
                frame_display = agregar_anotaciones_video(
                    frame, mascara, num_defectos, tiene_defectos, tiempo_inferencia, fps_actual
                )
            else:
                frame_display = frame.copy()
                cv2.putText(frame_display, "ERROR EN SEGMENTACION", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Agregar indicador de grabación si está activo
            if grabando_video:
                h, w = frame_display.shape[:2]
                cv2.circle(frame_display, (w-50, 50), 15, (0, 0, 255), -1)  # Círculo rojo
                cv2.putText(frame_display, "REC", (w-80, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
            
            # Grabar frame al video si está grabando y ha pasado suficiente tiempo
            if grabando_video and video_writer is not None:
                tiempo_actual = time.time()
                if tiempo_actual - tiempo_ultimo_frame_grabado >= intervalo_frame:
                    try:
                        if video_writer.isOpened():
                            video_writer.write(frame_display)
                            tiempo_ultimo_frame_grabado = tiempo_actual
                        else:
                            print("⚠️ VideoWriter se cerró inesperadamente")
                            grabando_video = False
                            video_writer = None
                    except Exception as e:
                        print(f"⚠️ Error escribiendo frame al video: {e}")
                        # Si hay error, detener grabación para evitar más problemas
                        grabando_video = False
                        try:
                            if video_writer:
                                video_writer.release()
                        except:
                            pass
                        video_writer = None
            
            # Mostrar frame
            cv2.imshow('Video Inferencia Coples', frame_display)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Saliendo...")
                break
            elif key == ord('s'):
                # Guardar frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                estado_str = "NG" if tiene_defectos else "OK"
                filename = f"cople_video_{timestamp}_{estado_str}_def{num_defectos}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                cv2.imwrite(filepath, frame_display)
                print(f"📸 Frame guardado: {filename}")
            elif key == ord('r'):
                # Iniciar/Detener grabación de video
                if not grabando_video:
                    # Iniciar grabación
                    video_writer, video_filename = iniciar_grabacion_video(output_dir, fps_grabacion, codec_preferido)
                    if video_writer is not None:
                        grabando_video = True
                        tiempo_ultimo_frame_grabado = time.time()
                else:
                    # Detener grabación
                    if detener_grabacion_video(video_writer, video_filename):
                        grabando_video = False
                        video_writer = None
                        video_filename = None
            elif key == ord('c'):
                # Cambiar codec preferido
                codecs_disponibles = ['MJPG', 'XVID', 'H264', 'MP4V']
                indice_actual = codecs_disponibles.index(codec_preferido) if codec_preferido in codecs_disponibles else 0
                nuevo_indice = (indice_actual + 1) % len(codecs_disponibles)
                codec_preferido = codecs_disponibles[nuevo_indice]
                print(f"🎞️  Codec cambiado a: {codec_preferido}")
                if grabando_video:
                    print("   ⚠️ El cambio se aplicará en la próxima grabación")
            
            # Mostrar estadísticas cada 30 frames
            if frame_count % 30 == 0:
                estado_str = "NG" if tiene_defectos else "OK"
                print(f"Frame #{frame_count:4d} | {estado_str:2s} | Def: {num_defectos:2d} | "
                      f"FPS: {fps_actual:4.1f} | Inf: {tiempo_inferencia:6.1f}ms")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
    
    finally:
        print("\n🧹 Liberando recursos...")
        
        try:
            # 1. PRIMERO: Detener grabación de video si está activa
            if grabando_video and video_writer is not None:
                try:
                    print("🎬 Deteniendo grabación de video...")
                    detener_grabacion_video(video_writer, video_filename)
                    video_writer = None  # Limpiar referencia
                    time.sleep(0.2)  # Pausa adicional
                except Exception as e:
                    print(f"⚠️ Error cerrando video writer: {e}")
                    try:
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                            time.sleep(0.2)
                    except:
                        pass
            
            # 2. SEGUNDO: Pausa antes de cerrar OpenCV
            time.sleep(0.1)
            
            # 3. TERCERO: Cerrar ventanas OpenCV de forma segura
            try:
                cv2.destroyAllWindows()
                time.sleep(0.1)
                cv2.waitKey(1)  # Procesar eventos de cierre
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Error cerrando ventanas OpenCV: {e}")
            
            # 4. CUARTO: Pausa antes de liberar cámara
            time.sleep(0.2)
            
            # 5. QUINTO: Liberar cámara de forma segura
            try:
                camara.liberar()
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Error liberando cámara: {e}")
            
            # 6. SEXTO: Limpiar memoria final
            try:
                import gc
                gc.collect()
                time.sleep(0.1)
            except:
                pass
                
            print(f"✅ Video terminado - {frame_count} frames procesados")
            
        except Exception as e:
            print(f"❌ Error en liberación de recursos: {e}")
            # Intentar salida forzada y limpia
            try:
                import sys
                sys.exit(0)
            except:
                pass

if __name__ == "__main__":
    main()
