"""
Configuración del sistema de captura y segmentación de coples
Contiene todas las constantes y parámetros configurables
"""

import os

# ==================== CONFIGURACIÓN DE CÁMARA ====================
class CameraConfig:
    """Configuración de la cámara GigE"""
    
    # Conexión
    DEFAULT_IP = "169.254.8.46"
    MAX_CAMERAS = 16
    
    # Parámetros de captura optimizados para resolución alta
    EXPOSURE_TIME = 15000      # 15ms - tiempo de exposición optimizado
    FRAMERATE = 10.0          # 10 FPS - reducido para menor carga CPU
    PACKET_SIZE = 9000        # Tamaño de paquete jumbo
    NUM_BUFFERS = 2           # Solo 2 buffers para minimizar memoria
    GAIN = 1.0               # Ganancia mínima para mejor calidad
    
    # Configuración del ROI
    ROI_WIDTH = 1280
    ROI_HEIGHT = 1024
    ROI_OFFSET_X = 1416
    ROI_OFFSET_Y = 576
    
    # Timeouts (en segundos)
    FRAME_TIMEOUT = 0.1       # 100ms timeout para frames
    STARTUP_TIMEOUT = 5.0     # 5s timeout para primer frame
    SHUTDOWN_TIMEOUT = 2.0    # 2s timeout para cerrar thread

# ==================== CONFIGURACIÓN DE INFERENCIA ====================
class InferenceConfig:
    """Configuración del modelo de inferencia ONNX"""
    
    # Archivos del modelo
    DEFAULT_MODEL_PATH = "coples_seg1C4V.onnx"
    CLASSES_FILE = "coples_seg_clases.txt"
    
    # Parámetros de preprocesamiento
    INPUT_SIZE = 1024         # Resolución del modelo (1024x1024)
    
    # Parámetros de inferencia
    CONFIDENCE_THRESHOLD = 0.5
    MAX_DETECTIONS = 3        # Máximo 3 detecciones para evitar problemas de memoria
    IOU_THRESHOLD = 0.5       # Para filtrar detecciones solapadas
    
    # Timeouts (en segundos)
    INFERENCE_TIMEOUT = 2.0   # 2s timeout para inferencia
    MASK_PROCESSING_TIMEOUT = 1.0  # 1s timeout para procesamiento de máscaras
    
    # Configuración ONNX
    INTRA_OP_THREADS = 2
    INTER_OP_THREADS = 2
    PROVIDERS = ['CPUExecutionProvider']

# ==================== CONFIGURACIÓN DE VISUALIZACIÓN ====================
class VisualizationConfig:
    """Configuración de visualización y colores"""
    
    # Colores (BGR format)
    DEFECT_COLOR = (0, 0, 255)        # Rojo para defectos
    CONTOUR_OUTER_COLOR = (0, 255, 255)  # Amarillo para contorno exterior
    CONTOUR_INNER_COLOR = (255, 255, 255)  # Blanco para contorno interior
    CENTER_POINT_COLOR = (0, 255, 0)    # Verde para puntos centrales
    TEXT_COLOR = (0, 255, 0)            # Verde para texto
    TIME_TEXT_COLOR = (255, 255, 0)     # Amarillo para tiempos
    
    # Parámetros de visualización
    OVERLAY_ALPHA = 0.35      # Transparencia del overlay
    CONTOUR_THICKNESS_OUTER = 3
    CONTOUR_THICKNESS_INNER = 1
    CENTER_POINT_RADIUS = 4
    MIN_CONTOUR_AREA = 100    # Área mínima para mostrar centro
    
    # Fuente de texto
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    SMALL_FONT_SCALE = 0.5
    SMALL_FONT_THICKNESS = 1

# ==================== CONFIGURACIÓN DE ARCHIVOS ====================
class FileConfig:
    """Configuración de archivos y directorios"""
    
    # Directorios
    OUTPUT_DIR = "Salida_cople"
    
    # Formatos de archivo
    IMAGE_FORMAT = ".jpg"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    # Nombres de archivo
    FILENAME_TEMPLATE = "cople_segmentacion_{timestamp}_#{count}{ext}"

# ==================== CONFIGURACIÓN DE ESTADÍSTICAS ====================
class StatsConfig:
    """Configuración de estadísticas y métricas"""
    
    # Tamaños de cola para estadísticas
    CAPTURE_TIMES_QUEUE_SIZE = 100
    PROCESSING_TIMES_QUEUE_SIZE = 100
    
    # Semilla para colores consistentes
    COLOR_SEED = 42

# ==================== CONFIGURACIÓN GLOBAL ====================
class GlobalConfig:
    """Configuración global del sistema"""
    
    # Control de procesamiento
    PROCESSING_ACTIVE = True
    USE_ELLIPTICAL_MASKS = True  # Usar máscaras elípticas (True) o rectangulares (False)
    
    # Configuraciones de sistema
    GIGEV_COMMON_PATH = "../gigev_common"
    
    # Límites de memoria
    BUFFER_MARGIN = 8192  # Margen extra para buffers
    
    @staticmethod
    def get_model_path():
        """Obtiene la ruta del modelo, verifica que existe"""
        if os.path.exists(InferenceConfig.DEFAULT_MODEL_PATH):
            return InferenceConfig.DEFAULT_MODEL_PATH
        else:
            raise FileNotFoundError(f"No se encontró el modelo: {InferenceConfig.DEFAULT_MODEL_PATH}")
    
    @staticmethod
    def get_classes_file():
        """Obtiene la ruta del archivo de clases, verifica que existe"""
        if os.path.exists(InferenceConfig.CLASSES_FILE):
            return InferenceConfig.CLASSES_FILE
        else:
            raise FileNotFoundError(f"No se encontró el archivo de clases: {InferenceConfig.CLASSES_FILE}")
    
    @staticmethod
    def ensure_output_dir():
        """Asegura que el directorio de salida existe"""
        if not os.path.exists(FileConfig.OUTPUT_DIR):
            os.makedirs(FileConfig.OUTPUT_DIR)
            print(f"Directorio '{FileConfig.OUTPUT_DIR}' creado")

# ==================== CONFIGURACIÓN DE DESARROLLO ====================
class DevConfig:
    """Configuración para desarrollo y debug"""
    
    # Flags de debug
    DEBUG_INFERENCE = False
    DEBUG_CAMERA = False
    DEBUG_PROCESSING = False
    
    # Límites de logging
    MAX_LOG_LINES = 1000
    
    # Configuración de logging
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR 