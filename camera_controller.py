"""
Controlador de cámara GigE para captura de imágenes de coples
Maneja la configuración, captura y buffering de imágenes de alta resolución
"""

import cv2
import time
import numpy as np
import ctypes
import threading
from threading import Event, Lock
from queue import Queue
import sys
import os

# Importar configuración y utilidades
from config import CameraConfig, StatsConfig, GlobalConfig
from utils import ipAddr_from_string, limpiar_memoria, log_performance, format_time_stats

# Obtener el código de soporte común para el GigE-V Framework
sys.path.append(os.path.dirname(__file__) + "/" + GlobalConfig.GIGEV_COMMON_PATH)

import pygigev
from pygigev import GevPixelFormats as GPF


class CamaraTiempoOptimizada:
    """
    Controlador optimizado de cámara GigE para captura de imágenes de coples.
    
    Características:
    - Captura asíncrona continua con doble buffer
    - Optimizado para resolución alta (1280x1024)
    - Procesamiento en tiempo real con mínima latencia
    - Gestión automática de memoria
    - Estadísticas de rendimiento en tiempo real
    """
    
    def __init__(self, ip=None):
        """
        Inicializa el controlador de cámara.
        
        Args:
            ip (str, optional): Dirección IP de la cámara. Si no se proporciona, usa la configuración por defecto.
        """
        self.ip = ip or CameraConfig.DEFAULT_IP
        self.handle = None
        self.buffer_addresses = None
        self.frame_count = 0
        self.camIndex = -1
        
        # Parámetros de configuración desde config.py
        self.exposure_time = CameraConfig.EXPOSURE_TIME
        self.framerate = CameraConfig.FRAMERATE
        self.packet_size = CameraConfig.PACKET_SIZE
        self.num_buffers = CameraConfig.NUM_BUFFERS
        self.gain = CameraConfig.GAIN
        
        # Configuración del ROI
        self.roi_width = CameraConfig.ROI_WIDTH
        self.roi_height = CameraConfig.ROI_HEIGHT
        self.roi_offset_x = CameraConfig.ROI_OFFSET_X
        self.roi_offset_y = CameraConfig.ROI_OFFSET_Y
        
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
        self.capture_times = Queue(maxsize=StatsConfig.CAPTURE_TIMES_QUEUE_SIZE)
        self.processing_times = Queue(maxsize=StatsConfig.PROCESSING_TIMES_QUEUE_SIZE)
        self.total_frames_captured = 0
        self.start_time = 0
        
        # Información de payload
        self.payload_size = None
        self.pixel_format = None

    def configurar_camara(self):
        """
        Configura parámetros de la cámara una sola vez.
        
        Returns:
            bool: True si la configuración fue exitosa
        """
        try:
            # Inicializar API GigE
            pygigev.GevApiInitialize()
            
            # Buscar cámaras disponibles
            numFound = (ctypes.c_uint32)(0)
            camera_info = (pygigev.GEV_CAMERA_INFO * CameraConfig.MAX_CAMERAS)()
            status = pygigev.GevGetCameraList(camera_info, CameraConfig.MAX_CAMERAS, ctypes.byref(numFound))
            
            if status != 0 or numFound.value == 0:
                print("❌ Error buscando cámaras")
                return False

            # Buscar cámara por IP
            target_ip_int = ipAddr_from_string(self.ip)
            self.camIndex = -1
            for i in range(numFound.value):
                if camera_info[i].ipAddr == target_ip_int:
                    self.camIndex = i
                    break

            if self.camIndex == -1:
                print(f"❗No se encontró la cámara con IP {self.ip}")
                return False

            # Abrir cámara
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
            if not self._configurar_parametros_camara():
                return False
            
            # Configurar ROI
            if not self._configurar_roi():
                return False
            
            # Configurar buffers
            if not self._configurar_buffers():
                return False
            
            # Inicializar transferencia asíncrona
            if not self._inicializar_transferencia():
                return False

            print("📷 Cámara configurada para captura asíncrona")
            return True

        except Exception as e:
            print(f"❌ Error en configuración: {e}")
            return False

    def _configurar_parametros_camara(self):
        """Configura los parámetros básicos de la cámara."""
        try:
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
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error configurando parámetros: {e}")
            return False

    def _configurar_roi(self):
        """Configura la región de interés (ROI)."""
        try:
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
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error configurando ROI: {e}")
            return False

    def _configurar_buffers(self):
        """Configura los buffers de captura."""
        try:
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
            bufsize = self.payload_size.value + GlobalConfig.BUFFER_MARGIN
            
            for i in range(self.num_buffers):
                temp = ((ctypes.c_char) * bufsize)()
                self.buffer_addresses[i] = ctypes.cast(temp, ctypes.c_void_p)

            print(f"✅ Buffers asignados: {self.num_buffers} de {bufsize} bytes")
            return True
            
        except Exception as e:
            print(f"❌ Error configurando buffers: {e}")
            return False

    def _inicializar_transferencia(self):
        """Inicializa la transferencia asíncrona."""
        try:
            status = pygigev.GevInitializeTransfer(
                self.handle,
                pygigev.Asynchronous,  # Modo asíncrono
                self.payload_size,
                self.num_buffers,
                self.buffer_addresses
            )
            if status != 0:
                print("❌ Error inicializando transferencia asíncrona")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando transferencia: {e}")
            return False

    def iniciar_captura_continua(self):
        """
        Inicia el thread de captura continua.
        
        Returns:
            bool: True si la captura se inició correctamente
        """
        if self.capture_thread and self.capture_thread.is_alive():
            print("⚠️ La captura ya está activa")
            return True
            
        self.capture_active = True
        self.capture_thread = threading.Thread(
            target=self._thread_captura_continua,
            daemon=True
        )
        self.capture_thread.start()
        
        # Esperar a que el primer frame esté listo
        if self.frame_ready_event.wait(timeout=CameraConfig.STARTUP_TIMEOUT):
            print("✅ Captura continua iniciada correctamente")
            return True
        else:
            print("❌ Timeout esperando primer frame")
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
                
                # Esperar frame con timeout
                status = pygigev.GevWaitForNextFrame(
                    self.handle,
                    ctypes.byref(gevbufPtr),
                    int(CameraConfig.FRAME_TIMEOUT * 1000)  # Convertir a ms
                )

                if status != 0:
                    if self.capture_active:
                        continue  # Timeout normal, continuar
                    else:
                        break

                capture_time = (time.time() - capture_start) * 1000
                
                # Procesar frame de manera asíncrona
                processing_start = time.time()
                if self._procesar_frame_async(gevbufPtr):
                    frame_local_count += 1
                    self.total_frames_captured += 1
                    
                    # Actualizar estadísticas
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
        """
        Procesa frame de manera asíncrona en el buffer de escritura actual.
        
        Args:
            gevbufPtr: Puntero al buffer de GigE
            
        Returns:
            bool: True si el procesamiento fue exitoso
        """
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

    def obtener_frame_instantaneo(self):
        """
        Obtiene el frame más reciente de manera instantánea (~1ms).
        
        Returns:
            tuple: (frame, tiempo_acceso_ms, timestamp) o (None, tiempo_acceso_ms, 0)
        """
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
        """
        Obtiene estadísticas de rendimiento.
        
        Returns:
            dict: Diccionario con estadísticas de rendimiento
        """
        if self.start_time == 0:
            return {}
            
        tiempo_total = time.time() - self.start_time
        fps_real = self.total_frames_captured / tiempo_total if tiempo_total > 0 else 0
        
        # Promedios de tiempos
        capture_times_list = list(self.capture_times.queue)
        processing_times_list = list(self.processing_times.queue)
        
        capture_stats = format_time_stats(capture_times_list)
        processing_stats = format_time_stats(processing_times_list)
        
        stats = {
            'fps_real': fps_real,
            'frames_totales': self.total_frames_captured,
            'tiempo_total': tiempo_total,
            'tiempo_captura': capture_stats,
            'tiempo_procesamiento': processing_stats,
            'buffers_listos': sum(self.frame_ready),
            'ip_camara': self.ip,
            'roi_size': f"{self.roi_width}x{self.roi_height}",
            'exposure_time': self.exposure_time,
            'framerate': self.framerate
        }
        
        return stats

    def detener_captura(self):
        """Detiene la captura continua."""
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=CameraConfig.SHUTDOWN_TIMEOUT)
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
            limpiar_memoria()
            
            print("✅ Recursos de cámara liberados correctamente")
            
        except Exception as e:
            print(f"❌ Error liberando recursos de cámara: {e}")

    def mostrar_configuracion(self):
        """Muestra la configuración actual de la cámara."""
        print(f"\n📷 CONFIGURACIÓN DE CÁMARA:")
        print(f"   IP: {self.ip}")
        print(f"   ROI: {self.roi_width}x{self.roi_height} @ ({self.roi_offset_x},{self.roi_offset_y})")
        print(f"   Exposición: {self.exposure_time} µs")
        print(f"   Frame Rate: {self.framerate} FPS")
        print(f"   Ganancia: {self.gain}")
        print(f"   Buffers: {self.num_buffers}")
        print(f"   Payload: {self.payload_size.value if self.payload_size else 'N/A'} bytes")
        
        if self.capture_active:
            print("   Estado: CAPTURANDO")
        else:
            print("   Estado: DETENIDO") 