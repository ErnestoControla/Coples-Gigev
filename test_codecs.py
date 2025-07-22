#!/usr/bin/env python3
"""
Script de prueba para verificar qué codecs de video están disponibles
en OpenCV para el sistema de grabación de coples.
"""

import cv2
import numpy as np
import os
import tempfile
from datetime import datetime

def test_codec(codec_name, fourcc_str, test_duration=2):
    """Prueba un codec específico"""
    print(f"\n🧪 Probando codec: {codec_name} ({fourcc_str})")
    
    try:
        # Crear archivo temporal
        temp_dir = tempfile.gettempdir()
        
        # Elegir extensión según codec
        if codec_name in ['H264', 'X264']:
            extension = '.mp4'
        else:
            extension = '.avi'
            
        test_file = os.path.join(temp_dir, f"test_{codec_name}_{datetime.now().strftime('%H%M%S')}{extension}")
        
        # Configurar VideoWriter con FPS apropiado para cada codec
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        
        # Algunos codecs tienen restricciones de FPS
        if codec_name == 'PIM1':  # MPEG-1 necesita FPS estándar
            fps = 25.0
        else:
            fps = 5.0
            
        frame_size = (640, 480)  # Tamaño más pequeño para prueba
        
        writer = cv2.VideoWriter(test_file, fourcc, fps, frame_size)
        
        if not writer.isOpened():
            print(f"   ❌ No se pudo crear VideoWriter con {extension}")
            # Intentar con AVI si estábamos usando MP4
            if extension == '.mp4':
                test_file_avi = test_file.replace('.mp4', '.avi')
                writer = cv2.VideoWriter(test_file_avi, fourcc, fps, frame_size)
                if writer.isOpened():
                    test_file = test_file_avi
                    print(f"   ✨ Funcionó con .avi en lugar de .mp4")
                else:
                    return False
            else:
                return False
        
        # Escribir algunos frames de prueba
        num_frames = int(fps * test_duration)  # Convertir a entero
        for i in range(num_frames):
            # Crear frame de prueba
            frame = np.random.randint(0, 255, (frame_size[1], frame_size[0], 3), dtype=np.uint8)
            
            # Agregar texto al frame
            cv2.putText(frame, f"Frame {i+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, codec_name, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            writer.write(frame)
        
        # Cerrar writer
        writer.release()
        
        # Verificar que el archivo existe y tiene contenido
        if not os.path.exists(test_file):
            print(f"   ❌ Archivo no creado")
            return False
        
        file_size = os.path.getsize(test_file)
        if file_size < 1024:  # Al menos 1KB
            print(f"   ❌ Archivo muy pequeño: {file_size} bytes")
            os.remove(test_file)
            return False
        
        # Intentar leer el video
        cap = cv2.VideoCapture(test_file)
        if not cap.isOpened():
            print(f"   ❌ No se puede abrir para lectura")
            os.remove(test_file)
            return False
        
        # Verificar propiedades
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        read_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Intentar leer un frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"   ❌ No se pudo leer frames")
            os.remove(test_file)
            return False
        
        print(f"   ✅ FUNCIONA CORRECTAMENTE")
        print(f"       📊 {frame_count} frames, {read_fps} FPS, {width}x{height}")
        print(f"       📁 Tamaño: {file_size/1024:.1f} KB")
        
        # Limpiar archivo temporal
        os.remove(test_file)
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        try:
            if os.path.exists(test_file):
                os.remove(test_file)
        except:
            pass
        return False

def main():
    """Función principal - prueba todos los codecs"""
    print("🎬 PRUEBA DE CODECS DE VIDEO PARA SISTEMA DE COPLES")
    print("=" * 55)
    
    # Lista de codecs a probar (ordenados por probabilidad de funcionar)
    codecs_to_test = [
        ('MJPG', 'MJPG'),  # Motion JPEG - MUY compatible
        ('XVID', 'XVID'),  # XVID - muy común en Linux
        ('MP4V', 'MP4V'),  # MPEG-4 - fallback común
        ('DIVX', 'DIVX'),  # DivX - antiguo pero funciona
        ('H264', 'H264'),  # H.264 - moderno (necesita codecs)
        ('X264', 'X264'),  # X264 - alternativa H.264
        ('FMP4', 'FMP4'),  # FFMPEG MP4
        ('PIM1', 'PIM1'),  # MPEG-1 - con FPS especial
    ]
    
    # Resultados
    codecs_working = []
    codecs_failed = []
    
    print(f"🔍 Probando {len(codecs_to_test)} codecs...")
    print(f"⏱️  Cada prueba genera ~2 segundos de video")
    print()
    
    for codec_name, fourcc_str in codecs_to_test:
        if test_codec(codec_name, fourcc_str):
            codecs_working.append(codec_name)
        else:
            codecs_failed.append(codec_name)
    
    # Mostrar resumen
    print("\n" + "=" * 55)
    print("📊 RESUMEN DE CODECS")
    print("=" * 55)
    
    if codecs_working:
        print(f"✅ CODECS QUE FUNCIONAN ({len(codecs_working)}):")
        for codec in codecs_working:
            print(f"   - {codec}")
        
        print(f"\n💡 RECOMENDACIÓN:")
        if 'MJPG' in codecs_working:
            print(f"   🥇 Usar MJPG (Motion JPEG) - muy compatible")
        elif 'XVID' in codecs_working:
            print(f"   🥈 Usar XVID - buena compatibilidad")
        else:
            print(f"   🥉 Usar {codecs_working[0]} - mejor opción disponible")
    else:
        print("❌ NINGÚN CODEC FUNCIONÓ CORRECTAMENTE")
        print("   Revisa la instalación de OpenCV")
    
    if codecs_failed:
        print(f"\n❌ CODECS QUE FALLARON ({len(codecs_failed)}):")
        for codec in codecs_failed:
            print(f"   - {codec}")
    
    print("\n🎯 Para el sistema de video de coples:")
    if codecs_working:
        print("   - Usa los codecs que funcionaron en la prueba")
        print("   - MJPG es el más universal y confiable")
        print("   - AVI es mejor formato que MP4 para estos codecs")
    else:
        print("   🚨 NINGÚN CODEC FUNCIONA - Instalar codecs:")
        print("   sudo apt update")
        print("   sudo apt install ffmpeg")
        print("   sudo apt install libx264-dev libxvidcore-dev")
        print("   sudo apt install ubuntu-restricted-extras")
        print("   ")
        print("   🔄 Después de instalar, ejecuta este script de nuevo")
        print("   ")
        print("   📱 Como alternativa temporal:")
        print("   - El sistema guardará frames individuales (tecla 's')")
        print("   - Puedes crear videos manualmente con:")
        print("     ffmpeg -r 5 -i cople_video_%06d.jpg -c:v libx264 video.mp4")

if __name__ == "__main__":
    main() 