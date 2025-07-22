#!/usr/bin/env python3
"""
Script de ejecución para el sistema de inferencia de video de coples
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(__file__))

# Importar y ejecutar el sistema de video
try:
    from cople_video.Inferencia_video import main
    
    print("🚀 Iniciando sistema de inferencia de video de coples...")
    print("   Modelo: coples_seg1C8V.onnx")
    print("   Cámara: 169.254.8.46")
    print("   Salida: salida_video/")
    print()
    
    main()
    
except ImportError as e:
    print(f"❌ Error importando el sistema de video: {e}")
    print("   Asegúrate de que el directorio cople_video existe")
    
except Exception as e:
    print(f"❌ Error ejecutando el sistema: {e}") 