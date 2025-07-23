"""
Sistema de Captura y Segmentación YOLO de Coples - Aplicación Principal
Integra todos los módulos del sistema modular para captura y análisis de imágenes
"""

import cv2
import time
import os
import numpy as np

# Importar módulos propios
from config import GlobalConfig, FileConfig
from utils import (
    verificar_dependencias, 
    mostrar_info_sistema,
    generar_nombre_archivo,
    calcular_estadisticas_mascara,
    limpiar_memoria
)
from camera_controller import CamaraTiempoOptimizada
from inference_engine import SegmentadorCoplesONNX
from image_processor import agregar_anotaciones_imagen


class SistemaSegmentacionCoples:
    """
    Sistema principal de captura y segmentación de coples.
    
    Integra el controlador de cámara y el motor de inferencia para proporcionar
    una interfaz completa de captura y análisis de imágenes.
    """
    
    def __init__(self, ip_camara=None, modelo_path=None):
        """
        Inicializa el sistema completo.
        
        Args:
            ip_camara (str, optional): IP de la cámara
            modelo_path (str, optional): Ruta del modelo ONNX
        """
        self.camara = CamaraTiempoOptimizada(ip=ip_camara)
        self.segmentador = SegmentadorCoplesONNX(model_path=modelo_path)
        self.frame_count = 0
        self.inicializado = False
        
        # Asegurar que el directorio de salida existe
        GlobalConfig.ensure_output_dir()
    
    def inicializar(self):
        """
        Inicializa todos los componentes del sistema.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        print("🚀 Inicializando sistema de segmentación de coples...")
        
        # Verificar dependencias
        if not verificar_dependencias():
            return False
        
        # Configurar la cámara
        print("\n📷 Configurando cámara...")
        if not self.camara.configurar_camara():
            print("❌ Error configurando la cámara")
            return False
        
        # Inicializar segmentador
        print("\n🧠 Inicializando motor de inferencia...")
        if not self.segmentador.inicializar():
            print("❌ Error inicializando segmentador de coples ONNX")
            return False
        
        # Iniciar captura continua
        print("\n🎯 Iniciando captura continua...")
        if not self.camara.iniciar_captura_continua():
            print("❌ Error iniciando captura continua")
            return False
        
        self.inicializado = True
        print("✅ Sistema inicializado correctamente")
        return True
    
    def capturar_y_segmentar(self):
        """
        Captura una imagen y la segmenta con el modelo YOLO.
        
        Returns:
            tuple: (frame, mascara, imagen_coloreada, tiempo_captura, tiempo_inferencia, tiempo_total)
        """
        if not self.inicializado:
            return None, None, None, 0, 0, 0
        
        start_total = time.time()
        
        # Capturar frame
        start_capture = time.time()
        frame, tiempo_acceso, timestamp = self.camara.obtener_frame_instantaneo()
        tiempo_captura = (time.time() - start_capture) * 1000
        
        if frame is None:
            tiempo_total = (time.time() - start_total) * 1000
            return None, None, None, tiempo_captura, 0, tiempo_total
        
        # Segmentar imagen
        mascara, imagen_coloreada, tiempo_inferencia = self.segmentador.segmentar(frame)
        
        tiempo_total = (time.time() - start_total) * 1000
        return frame, mascara, imagen_coloreada, tiempo_captura, tiempo_inferencia, tiempo_total
    
    def obtener_frame_simple(self):
        """
        Obtiene un frame simple sin segmentación.
        
        Returns:
            tuple: (frame, tiempo_acceso, timestamp)
        """
        if not self.inicializado:
            return None, 0, 0
        
        return self.camara.obtener_frame_instantaneo()
    
    def guardar_imagen(self, imagen, mascara=None, imagen_original=None):
        """
        Guarda una imagen con anotaciones en el directorio de salida.
        
        Args:
            imagen (np.ndarray): Imagen procesada a guardar (con anotaciones/máscaras)
            mascara (np.ndarray, optional): Máscara de segmentación
            imagen_original (np.ndarray, optional): Imagen original sin procesar
            
        Returns:
            str: Ruta del archivo guardado o None si hay error
        """
        try:
            # Generar nombre de archivo
            timestamp = time.strftime(FileConfig.TIMESTAMP_FORMAT)
            self.frame_count += 1
            filename = generar_nombre_archivo(timestamp, self.frame_count)
            filepath = os.path.join(FileConfig.OUTPUT_DIR, filename)
            
            # Guardar imagen procesada
            cv2.imwrite(filepath, imagen)
            print(f"📁 Imagen procesada guardada: {filepath}")
            
            # Guardar imagen original si se proporciona
            if imagen_original is not None:
                base_name = os.path.splitext(filepath)[0]
                original_path = f"{base_name}_original.jpg"
                cv2.imwrite(original_path, imagen_original)
                print(f"📁 Imagen original guardada: {original_path}")
            
            # Mostrar estadísticas si hay máscara
            if mascara is not None:
                stats = calcular_estadisticas_mascara(mascara)
                if stats['defect_pixels'] > 0:
                    print(f"   📊 Defectos: {stats['defect_pixels']} píxeles ({stats['percentage']:.2f}%)")
                
                # Guardar resultados intermedios para desarrollo
                from config import DevConfig
                if DevConfig.SAVE_INTERMEDIATE_RESULTS and stats['defect_pixels'] > 0:
                    self._guardar_resultados_desarrollo(filepath, mascara, stats, imagen_original)
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error guardando imagen: {e}")
            return None
    
    def _guardar_resultados_desarrollo(self, filepath_base, mascara, stats, imagen_original=None):
        """
        Guarda resultados intermedios para análisis de desarrollo.
        
        Args:
            filepath_base (str): Ruta base del archivo
            mascara (np.ndarray): Máscara de segmentación
            stats (dict): Estadísticas de la máscara
            imagen_original (np.ndarray, optional): Imagen original sin procesar
        """
        try:
            import json
            
            # Crear nombre base sin extensión
            base_name = os.path.splitext(filepath_base)[0]
            
            # Guardar máscara como imagen separada
            mask_path = f"{base_name}_mask.png"
            # Convertir máscara a escala de grises visible
            mask_visual = (mascara * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_visual)
            
            # Guardar estadísticas como JSON
            stats_path = f"{base_name}_stats.json"
            stats_to_save = {
                'total_pixels': int(stats['total_pixels']),
                'defect_pixels': int(stats['defect_pixels']),
                'percentage': float(stats['percentage']),
                'num_regions': int(stats['num_regions']),
                'areas': [float(area) for area in stats['areas']],
                'avg_area': float(stats['avg_area']),
                'max_area': float(stats['max_area']),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(stats_path, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
            
            print(f"   🔍 Resultados debug guardados:")
            print(f"      - Máscara: {mask_path}")
            print(f"      - Estadísticas: {stats_path}")
            
        except Exception as e:
            print(f"   ⚠️ Error guardando resultados debug: {e}")
    
    def obtener_estadisticas(self):
        """
        Obtiene estadísticas completas del sistema.
        
        Returns:
            dict: Estadísticas del sistema
        """
        stats_camara = self.camara.obtener_estadisticas()
        stats_modelo = self.segmentador.obtener_info_modelo()
        
        return {
            'camara': stats_camara,
            'modelo': stats_modelo,
            'frames_procesados': self.frame_count,
            'sistema_inicializado': self.inicializado
        }
    
    def cambiar_tipo_mascara(self):
        """
        Cambia el tipo de máscara fallback.
        
        Returns:
            str: Tipo de máscara actual
        """
        return self.segmentador.cambiar_tipo_mascara()
    
    def mostrar_configuracion(self):
        """Muestra la configuración completa del sistema."""
        print("\n" + "="*70)
        print("📋 CONFIGURACIÓN DEL SISTEMA")
        print("="*70)
        
        # Configuración de cámara
        self.camara.mostrar_configuracion()
        
        # Configuración del modelo
        self.segmentador.mostrar_configuracion()
        
        print("="*70)
    
    def liberar(self):
        """Libera todos los recursos del sistema."""
        print("\n🧹 Liberando recursos del sistema...")
        
        try:
            # Liberar cámara
            self.camara.liberar()
            
            # Limpiar memoria
            limpiar_memoria()
            
            print("✅ Recursos liberados correctamente")
            
        except Exception as e:
            print(f"❌ Error liberando recursos: {e}")


def mostrar_menu():
    """Muestra el menú de opciones disponibles."""
    print("\n" + "="*60)
    print("🎯 COMANDOS DISPONIBLES:")
    print("="*60)
    print("  ENTER - Capturar imagen y segmentar coples con YOLO")
    print("  'v'   - Solo ver frame (sin segmentar)")
    print("  's'   - Mostrar estadísticas del sistema")
    print("  'c'   - Mostrar configuración completa")
    print("  'm'   - Cambiar tipo de máscara fallback")
    print("  'q'   - Salir del sistema")
    print("="*60)


def procesar_comando_captura(sistema, ventana_cv):
    """
    Procesa el comando de captura y segmentación.
    
    Args:
        sistema (SistemaSegmentacionCoples): Sistema principal
        ventana_cv (str): Nombre de la ventana OpenCV
    """
    frame, mascara, imagen_coloreada, tiempo_captura, tiempo_inferencia, tiempo_total = sistema.capturar_y_segmentar()
    
    if frame is not None and mascara is not None:
        print(f"\n🔍 RESULTADO DE SEGMENTACIÓN YOLO #{sistema.frame_count}")
        print("=" * 60)
        print(f"⏱️  TIEMPOS:")
        print(f"   Captura:    {tiempo_captura:.2f} ms")
        print(f"   Inferencia: {tiempo_inferencia:.2f} ms")
        print(f"   Total:      {tiempo_total:.2f} ms")
        
        # Analizar segmentación
        clases_detectadas = np.unique(mascara)
        print(f"\n🎯 SEGMENTACIÓN YOLO:")
        
        if 1 in clases_detectadas:
            nombre_clase = sistema.segmentador.class_names[0] if sistema.segmentador.class_names else "Defecto"
            pixels_defecto = np.sum(mascara == 1)
            porcentaje = (pixels_defecto / mascara.size) * 100
            print(f"   - {nombre_clase}: {pixels_defecto} píxeles ({porcentaje:.2f}%)")
            print(f"   - Estado: DEFECTO SEGMENTADO")
        else:
            print(f"   - Estado: SIN DEFECTOS SEGMENTADOS")
        
        print("=" * 60)
        
        # Crear imagen con anotaciones
        if imagen_coloreada is not None:
            frame_anotado = imagen_coloreada.copy()
        else:
            frame_anotado = frame.copy()
        
        # Agregar anotaciones
        frame_anotado = agregar_anotaciones_imagen(
            frame_anotado, mascara, sistema.segmentador.class_names,
            tiempo_captura, tiempo_inferencia, tiempo_total,
            sistema.segmentador.usar_mascaras_elipticas
        )
        
        # Guardar imagen
        sistema.guardar_imagen(frame_anotado, mascara, frame)
        
        # Mostrar imagen
        cv2.imshow(ventana_cv, frame_anotado)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    else:
        print("⚠️ No hay frames disponibles o error en segmentación")
    
    return True


def procesar_comando_ver(sistema, ventana_cv):
    """
    Procesa el comando de ver frame sin segmentar.
    
    Args:
        sistema (SistemaSegmentacionCoples): Sistema principal
        ventana_cv (str): Nombre de la ventana OpenCV
    """
    frame, tiempo_acceso, timestamp = sistema.obtener_frame_simple()
    
    if frame is not None:
        print(f"📷 Frame obtenido en {tiempo_acceso:.2f} ms")
        
        # Mostrar frame
        cv2.imshow(ventana_cv, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    else:
        print("⚠️ No hay frames disponibles")
    
    return True


def procesar_comando_estadisticas(sistema):
    """
    Procesa el comando de mostrar estadísticas.
    
    Args:
        sistema (SistemaSegmentacionCoples): Sistema principal
    """
    stats = sistema.obtener_estadisticas()
    
    print(f"\n📊 ESTADÍSTICAS DEL SISTEMA:")
    print("=" * 50)
    
    # Estadísticas de cámara
    if stats['camara']:
        cam_stats = stats['camara']
        print(f"📷 CÁMARA:")
        print(f"   FPS Real: {cam_stats.get('fps_real', 0):.1f}")
        print(f"   Frames Totales: {cam_stats.get('frames_totales', 0)}")
        print(f"   Buffers Listos: {cam_stats.get('buffers_listos', 0)}/2")
        
        # Estadísticas de tiempo
        tiempo_cap = cam_stats.get('tiempo_captura', {})
        tiempo_proc = cam_stats.get('tiempo_procesamiento', {})
        if tiempo_cap:
            print(f"   Tiempo Captura: {tiempo_cap.get('promedio', 0):.2f} ms (±{tiempo_cap.get('std', 0):.2f})")
        if tiempo_proc:
            print(f"   Tiempo Procesamiento: {tiempo_proc.get('promedio', 0):.2f} ms (±{tiempo_proc.get('std', 0):.2f})")
    
    # Estadísticas del modelo
    if stats['modelo']:
        model_stats = stats['modelo']
        print(f"\n🧠 MODELO:")
        print(f"   Clases: {model_stats.get('num_clases', 0)}")
        print(f"   Procesamiento: {'ACTIVO' if model_stats.get('procesamiento_activo', False) else 'INACTIVO'}")
        print(f"   Máscaras: {'Elípticas' if model_stats.get('usar_mascaras_elipticas', False) else 'Rectangulares'}")
    
    print(f"\n📈 SISTEMA:")
    print(f"   Frames Procesados: {stats['frames_procesados']}")
    print(f"   Estado: {'OPERATIVO' if stats['sistema_inicializado'] else 'NO INICIALIZADO'}")
    print("=" * 50)


def main():
    """Función principal del sistema de segmentación de coples."""
    # Mostrar información del sistema
    mostrar_info_sistema()
    
    # Inicializar sistema
    sistema = SistemaSegmentacionCoples()
    
    if not sistema.inicializar():
        print("❌ Error inicializando el sistema")
        return
    
    # Mostrar menú inicial
    mostrar_menu()
    
    # Crear ventana OpenCV
    ventana_cv = 'Sistema de Segmentación YOLO de Coples'
    cv2.namedWindow(ventana_cv, cv2.WINDOW_NORMAL)
    
    try:
        # Bucle principal de la aplicación
        while True:
            entrada = input("\n🎯 Comando: ").strip().lower()
            
            if entrada == 'q':
                print("🔄 Saliendo del sistema...")
                break
            
            elif entrada == 's':
                procesar_comando_estadisticas(sistema)
            
            elif entrada == 'c':
                sistema.mostrar_configuracion()
            
            elif entrada == 'm':
                tipo_mascara = sistema.cambiar_tipo_mascara()
                print(f"   💡 Tipo actual: {tipo_mascara}")
            
            elif entrada == 'v':
                if not procesar_comando_ver(sistema, ventana_cv):
                    break
            
            elif entrada == '':
                # Comando de captura (ENTER)
                if not procesar_comando_captura(sistema, ventana_cv):
                    break
            
            elif entrada == 'help' or entrada == 'h':
                mostrar_menu()
            
            else:
                print("❓ Comando no reconocido. Escribe 'help' para ver opciones.")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
    
    finally:
        # Limpieza final
        print("\n🧹 Limpiando recursos...")
        try:
            sistema.liberar()
        except:
            pass
        
        # Limpiar OpenCV
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
        
        # Liberar memoria final
        limpiar_memoria()
        
        print("✅ Sistema terminado correctamente")


if __name__ == "__main__":
    main() 