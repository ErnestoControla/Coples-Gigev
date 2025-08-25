import sys
import ctypes
import time
import os
import cv2
import numpy as np
import threading
from threading import Lock
from datetime import datetime

# Obtener el código de soporte común para el GigE-V Framework
sys.path.append(os.path.dirname(__file__) + "/../gigev_common")

import pygigev
from pygigev import GevPixelFormats as GPF

def ipAddr_from_string(s):
    """Convertir dirección IPv4 con puntos a entero."""
    from functools import reduce
    return reduce(lambda a, b: a << 8 | b, map(int, s.split(".")))

class CamaraCompleta:
    def __init__(self, ip="172.16.1.21"):
        self.ip = ip
        self.handle = None
        self.buffer_addresses = None
        
        # Parámetros de configuración
        self.exposure_time = 10000   # 10ms 
        self.framerate = 30.0       # FPS más alto con ROI
        self.packet_size = 9000     # Tamaño de paquete jumbo
        self.num_buffers = 4        # 4 buffers para mejor flujo
        self.gain = 2.0
        
        # Configuración del ROI (tamaño manejable)
        self.roi_width = 1280
        self.roi_height = 1024
        self.roi_offset_x = 1416
        self.roi_offset_y = 576
        
        # Control de captura
        self.capture_active = False
        self.capture_thread = None
        self.frame_lock = Lock()
        self.current_frame = None

    def configurar_camara(self):
        """Configurar parámetros de la cámara con ROI."""
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

            # Leer tamaño final del ROI
            feature_strlen = ctypes.c_int(pygigev.MAX_GEVSTRING_LENGTH)
            unused = ctypes.c_int(0)
            width_str = (ctypes.c_char * feature_strlen.value)()
            height_str = (ctypes.c_char * feature_strlen.value)()
            pygigev.GevGetFeatureValueAsString(self.handle, b"Width", unused, feature_strlen, width_str)
            pygigev.GevGetFeatureValueAsString(self.handle, b"Height", unused, feature_strlen, height_str)
            self.width = int(width_str.value)
            self.height = int(height_str.value)
            print(f"📏 Tamaño de imagen ROI: {self.width}x{self.height}")

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

            # Configurar buffers
            self.buffer_addresses = ((ctypes.c_void_p) * self.num_buffers)()
            bufsize = self.payload_size.value + 8192
            
            for i in range(self.num_buffers):
                temp = ((ctypes.c_char) * bufsize)()
                self.buffer_addresses[i] = ctypes.cast(temp, ctypes.c_void_p)

            print(f"✅ Buffers asignados: {self.num_buffers} de {bufsize} bytes")

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

            print("📷 Cámara configurada correctamente")
            return True

        except Exception as e:
            print(f"❌ Error en configuración: {e}")
            return False

    def _thread_captura_continua(self):
        """Thread dedicado a captura continua de frames."""
        print("🚀 Iniciando captura continua...")
        
        status = pygigev.GevStartTransfer(self.handle, -1)
        if status != 0:
            print("❌ Error iniciando transferencia continua")
            return
            
        try:
            while self.capture_active:
                gevbufPtr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()
                
                status = pygigev.GevWaitForNextFrame(
                    self.handle,
                    ctypes.byref(gevbufPtr),
                    50
                )

                if status != 0:
                    if self.capture_active:
                        continue
                    else:
                        break

                gevbuf = gevbufPtr.contents
                if gevbuf.status == 0:
                    # Convertir datos del buffer
                    im_addr = ctypes.cast(
                        gevbuf.address,
                        ctypes.POINTER(ctypes.c_ubyte * gevbuf.recv_size)
                    )
                    raw_data = np.frombuffer(im_addr.contents, dtype=np.uint8)
                    raw_data = raw_data.reshape((self.height, self.width))
                    
                    # Procesar imagen
                    frame_rgb = cv2.cvtColor(raw_data, cv2.COLOR_BayerRG2RGB)
                    
                    # Actualizar frame actual
                    with self.frame_lock:
                        self.current_frame = frame_rgb.copy()
                
                # Liberar el buffer
                if gevbufPtr:
                    pygigev.GevReleaseFrame(self.handle, gevbufPtr)
                    
        except Exception as e:
            print(f"❌ Error en thread de captura: {e}")
        finally:
            if self.handle:
                pygigev.GevStopTransfer(self.handle)
            print("📊 Thread de captura terminado")

    def iniciar_captura(self):
        """Inicia la captura continua."""
        if self.capture_thread and self.capture_thread.is_alive():
            print("⚠️ La captura ya está activa")
            return True
            
        self.capture_active = True
        self.capture_thread = threading.Thread(
            target=self._thread_captura_continua,
            daemon=True
        )
        self.capture_thread.start()
        print("✅ Captura continua iniciada")
        return True

    def obtener_frame(self):
        """Obtiene el frame más reciente."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def detener_captura(self):
        """Detiene la captura continua."""
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        print("🛑 Captura continua detenida")

    def liberar(self):
        """Liberar recursos de la cámara."""
        try:
            self.detener_captura()
            
            if self.handle:
                pygigev.GevCloseCamera(self.handle)
                self.handle = None
            
            pygigev.GevApiUninitialize()
            print("✅ Recursos liberados correctamente")
            
        except Exception as e:
            print(f"❌ Error liberando recursos: {e}")

def main():
    """Función principal con menú de captura y salida."""
    camara = CamaraCompleta()
    
    if not camara.configurar_camara():
        print("❌ Error configurando la cámara")
        return

    if not camara.iniciar_captura():
        print("❌ Error iniciando captura")
        camara.liberar()
        return

    print("\n🎥 VISUALIZACIÓN EN TIEMPO REAL")
    print("Presione 'q' para salir | 'c' para capturar imagen")
    
    cv2.namedWindow('Cámara Completa', cv2.WINDOW_NORMAL)
    carpeta_capturas = os.path.join(os.path.dirname(__file__), "Capturas")
    if not os.path.exists(carpeta_capturas):
        os.makedirs(carpeta_capturas)
    
    try:
        # Precalcular la tabla de corrección gamma
        gamma = 1.5  # Ahora aclara la imagen
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        while True:
            frame = camara.obtener_frame()
            if frame is not None:
                # Aplicar corrección gamma
                frame_gamma = cv2.LUT(frame, table)
                cv2.imshow('Cámara Completa', frame_gamma)
            else:
                frame_gamma = None
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Saliendo...")
                break
            elif key == ord('c'):
                if frame_gamma is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    nombre_archivo = f"captura_{timestamp}.png"
                    ruta = os.path.join(carpeta_capturas, nombre_archivo)
                    cv2.imwrite(ruta, frame_gamma)
                    print(f"📸 Imagen capturada: {ruta}")
                else:
                    print("⚠️ No hay frame disponible para capturar")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
    
    finally:
        print("\n🧹 Liberando recursos...")
        camara.liberar()
        cv2.destroyAllWindows()
        print("✅ Aplicación terminada")

if __name__ == "__main__":
    main()
