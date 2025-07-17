#!/usr/bin/env python3
"""
Script de análisis para desarrollo - Interpretación de resultados del modelo
Ayuda a entender qué está detectando el modelo para mejorar el entrenamiento
"""

import cv2
import numpy as np
import json
import os
from main import SistemaSegmentacionCoples
from config import DevConfig

def analizar_resultado_detallado():
    """
    Ejecuta el sistema con análisis detallado activado.
    """
    print("🔍 MODO ANÁLISIS DETALLADO PARA DESARROLLO")
    print("=" * 60)
    print("Este modo proporciona información completa sobre:")
    print("- Detecciones individuales del modelo")
    print("- Procesamiento de máscaras paso a paso")
    print("- Estadísticas de conectividad")
    print("- Archivos debug adicionales")
    print("=" * 60)
    
    # Verificar que el modo debug está activado
    if not DevConfig.DEBUG_INFERENCE:
        print("⚠️  Activando modo DEBUG en configuración...")
        DevConfig.DEBUG_INFERENCE = True
        DevConfig.DEBUG_MASKS = True
        DevConfig.SAVE_INTERMEDIATE_RESULTS = True
    
    # Inicializar sistema
    sistema = SistemaSegmentacionCoples()
    
    if not sistema.inicializar():
        print("❌ Error inicializando el sistema")
        return
    
    # Mostrar configuración del modelo
    info_modelo = sistema.segmentador.obtener_info_modelo()
    print(f"\n📋 CONFIGURACIÓN DEL MODELO:")
    print(f"   - Modelo: {info_modelo['ruta_modelo']}")
    print(f"   - Clases: {info_modelo['clases']}")
    print(f"   - Umbral confianza: {info_modelo['umbral_confianza']}")
    print(f"   - Máx. detecciones: {info_modelo['max_detecciones']}")
    
    # Crear ventana
    cv2.namedWindow('Análisis Detallado', cv2.WINDOW_NORMAL)
    
    try:
        print(f"\n🎯 Comandos disponibles:")
        print(f"   ENTER - Capturar con análisis detallado")
        print(f"   'q' - Salir")
        
        frame_count = 0
        
        while True:
            entrada = input(f"\n🔍 Análisis #{frame_count + 1}: ").strip().lower()
            
            if entrada == 'q':
                break
            
            elif entrada == '':
                print(f"\n{'='*80}")
                print(f"🎯 ANÁLISIS DETALLADO #{frame_count + 1}")
                print(f"{'='*80}")
                
                # Capturar con análisis detallado
                frame, mascara, imagen_coloreada, t_cap, t_inf, t_total = sistema.capturar_y_segmentar()
                
                if frame is not None and mascara is not None:
                    frame_count += 1
                    
                    # Mostrar información resumida
                    print(f"\n📊 RESUMEN DEL ANÁLISIS:")
                    print(f"   ⏱️  Tiempo total: {t_total:.2f}ms")
                    print(f"   🎯 Defectos detectados: {np.sum(mascara == 1)} píxeles")
                    print(f"   📏 Resolución: {frame.shape[1]}x{frame.shape[0]}")
                    
                    # Guardar imagen
                    filepath = sistema.guardar_imagen(
                        imagen_coloreada if imagen_coloreada is not None else frame, 
                        mascara
                    )
                    
                    # Mostrar análisis de archivos generados
                    if filepath:
                        analizar_archivos_generados(filepath)
                    
                    # Mostrar imagen
                    if imagen_coloreada is not None:
                        cv2.imshow('Análisis Detallado', imagen_coloreada)
                        cv2.waitKey(1)
                    
                    print(f"\n{'='*80}")
                    
                else:
                    print("⚠️ No se pudo capturar o procesar la imagen")
    
    finally:
        sistema.liberar()
        cv2.destroyAllWindows()

def analizar_archivos_generados(filepath):
    """
    Analiza los archivos debug generados.
    
    Args:
        filepath (str): Ruta del archivo principal
    """
    base_name = os.path.splitext(filepath)[0]
    
    # Analizar estadísticas JSON
    stats_path = f"{base_name}_stats.json"
    if os.path.exists(stats_path):
        print(f"\n📊 ANÁLISIS DE ESTADÍSTICAS:")
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            print(f"   📐 Análisis de áreas:")
            if stats['areas']:
                areas_sorted = sorted(stats['areas'], reverse=True)
                for i, area in enumerate(areas_sorted[:5]):
                    porcentaje = (area / stats['total_pixels']) * 100
                    print(f"      - Región {i+1}: {area:.0f} píxeles ({porcentaje:.4f}%)")
            
            print(f"   📊 Distribución:")
            print(f"      - Total píxeles: {stats['total_pixels']:,}")
            print(f"      - Píxeles defectuosos: {stats['defect_pixels']:,}")
            print(f"      - Porcentaje: {stats['percentage']:.4f}%")
            print(f"      - Regiones: {stats['num_regions']}")
            print(f"      - Área promedio: {stats['avg_area']:.1f} píxeles")
            
        except Exception as e:
            print(f"   ❌ Error leyendo estadísticas: {e}")
    
    # Analizar máscara
    mask_path = f"{base_name}_mask.png"
    if os.path.exists(mask_path):
        print(f"\n🎭 ANÁLISIS DE MÁSCARA:")
        try:
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                unique_values = np.unique(mask_img)
                print(f"   📊 Valores en máscara: {unique_values}")
                
                # Análisis de conectividad
                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"   🔗 Componentes conectados: {len(contours)}")
                
                # Información de forma
                if contours:
                    print(f"   📐 Análisis de forma:")
                    for i, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        
                        if area > 0:
                            # Calcular métricas de forma
                            aspect_ratio = cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]
                            solidity = area / cv2.contourArea(cv2.convexHull(contour))
                            
                            print(f"      - Componente {i+1}: área={area:.0f}, perímetro={perimeter:.1f}")
                            print(f"        → Aspecto: {aspect_ratio:.2f}, Solidez: {solidity:.2f}")
                
        except Exception as e:
            print(f"   ❌ Error analizando máscara: {e}")

def interpretar_salida_modelo():
    """
    Ayuda a interpretar la salida específica del modelo.
    """
    print("\n📋 INTERPRETACIÓN DE SALIDA DEL MODELO")
    print("=" * 50)
    print()
    
    print("🔍 SIGNIFICADO DE LOS NÚMEROS:")
    print("   1. 'Detecciones encontradas: N' = Número de objetos detectados")
    print("   2. 'Máscara real generada: X píxeles' = Píxeles por detección individual")
    print("   3. 'Píxeles defectuosos: Y' = Total final combinado")
    print("   4. 'Regiones detectadas: Z' = Componentes conectados en máscara final")
    print()
    
    print("⚠️  POSIBLES INCONSISTENCIAS:")
    print("   - Detecciones = 3, pero píxeles no suman")
    print("   - Máscaras individuales pequeñas vs total grande")
    print("   - Regiones diferentes a detecciones")
    print()
    
    print("🎯 PARA MEJORAR EL ENTRENAMIENTO:")
    print("   1. Verificar que las detecciones corresponden a defectos reales")
    print("   2. Revisar si las máscaras siguen la forma correcta")
    print("   3. Validar que no hay superposición excesiva")
    print("   4. Confirmar que el umbral de confianza es adecuado")
    print()
    
    print("📊 USAR MODO DEBUG PARA VER:")
    print("   - Coordenadas exactas de cada detección")
    print("   - Coeficientes de máscara por detección")
    print("   - Estadísticas de conectividad")
    print("   - Archivos separados para análisis manual")

def modo_comparacion():
    """
    Modo para comparar diferentes configuraciones del modelo.
    """
    print("\n🔄 MODO COMPARACIÓN DE CONFIGURACIONES")
    print("=" * 50)
    
    configuraciones = [
        {
            'nombre': 'Actual',
            'modelo': 'coples_seg1C4V.onnx',
            'confianza': 0.5,
            'max_detecciones': 3
        },
        {
            'nombre': 'Menos estricto',
            'modelo': 'coples_seg1C4V.onnx',
            'confianza': 0.3,
            'max_detecciones': 5
        },
        {
            'nombre': 'Más estricto',
            'modelo': 'coples_seg1C4V.onnx',
            'confianza': 0.7,
            'max_detecciones': 2
        }
    ]
    
    print("📋 Configuraciones disponibles:")
    for i, config in enumerate(configuraciones):
        print(f"   {i+1}. {config['nombre']}: confianza={config['confianza']}, max_det={config['max_detecciones']}")
    
    print("\n💡 Para implementar: Modificar InferenceConfig en config.py")
    print("   - CONFIDENCE_THRESHOLD para cambiar umbral")
    print("   - MAX_DETECTIONS para cambiar límite")

if __name__ == "__main__":
    print("🔍 HERRAMIENTAS DE ANÁLISIS PARA DESARROLLO")
    print("=" * 50)
    print("1. Análisis detallado con debug")
    print("2. Interpretación de salida del modelo")
    print("3. Modo comparación de configuraciones")
    print("4. Salir")
    
    while True:
        opcion = input("\nElige opción (1-4): ").strip()
        
        if opcion == "1":
            analizar_resultado_detallado()
        elif opcion == "2":
            interpretar_salida_modelo()
        elif opcion == "3":
            modo_comparacion()
        elif opcion == "4":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❓ Opción no válida. Intenta de nuevo.") 