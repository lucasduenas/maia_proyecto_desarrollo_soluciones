#!/usr/bin/env python3
"""
MLflow Development Server - Sin validaciones de seguridad
SOLO PARA DESARROLLO - NO USAR EN PRODUCCIÓN

Este script inicia MLflow con todas las validaciones de seguridad deshabilitadas
para evitar problemas con CORS y Host headers durante el desarrollo.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    # Configuración
    PORT = 8050
    HOST = '0.0.0.0'
    
    # Rutas
    raiz = Path.cwd().resolve()
    if raiz.name == 'scripts':
        raiz = raiz.parent
    
    backend = str(raiz / 'outputs' / 'hiatal_mlflow' / 'mlruns')
    artifacts = str(raiz / 'outputs' / 'hiatal_mlflow' / 'mlartifacts')
    
    # Crear directorios
    Path(backend).mkdir(parents=True, exist_ok=True)
    Path(artifacts).mkdir(parents=True, exist_ok=True)
    
    # Banner
    print("\n" + "="*80)
    print(" "*20 + "🚀 MLflow Development Server")
    print("="*80)
    print(f"\n📍 Puerto: {PORT}")
    print(f"📁 Backend: {backend}")
    print(f"📦 Artifacts: {artifacts}")
    print(f"\n⚠️  MODO DESARROLLO - Todas las validaciones deshabilitadas")
    print("="*80 + "\n")
    
    # Configurar variables de entorno
    env = os.environ.copy()
    
    # CORS - Permitir TODO
    env.update({
        'MLFLOW_ENABLE_CORS': 'true',
        'MLFLOW_CORS_ALLOW_ORIGIN': '*',
        'MLFLOW_CORS_ALLOW_METHODS': '*',
        'MLFLOW_CORS_ALLOW_HEADERS': '*',
        'MLFLOW_CORS_ALLOW_CREDENTIALS': 'true',
        'MLFLOW_CORS_EXPOSE_HEADERS': '*',
        'MLFLOW_CORS_MAX_AGE': '86400',
        
        # Hosts - Permitir TODO
        'MLFLOW_ALLOWED_HOSTS': '*',
        
        # Deshabilitar TODAS las validaciones
        'MLFLOW_DISABLE_HOST_VALIDATION': 'true',
        'MLFLOW_DISABLE_CSRF_PROTECTION': 'true',
        'MLFLOW_TRACKING_INSECURE_TLS': 'true',
        
        # Configuración adicional
        'MLFLOW_SERVE_ARTIFACTS': 'true',
    })
    
    # Comando
    cmd = [
        sys.executable, '-m', 'mlflow', 'server',
        '--host', HOST,
        '--port', str(PORT),
        '--backend-store-uri', backend,
        '--default-artifact-root', artifacts,
        '--serve-artifacts',
        '--gunicorn-opts', '--timeout 300 --workers 1 --log-level warning',
    ]
    
    print("🔧 Configuración aplicada:")
    print("   ✓ CORS habilitado para todos los orígenes")
    print("   ✓ Validación de host deshabilitada")
    print("   ✓ Protección CSRF deshabilitada")
    print("   ✓ Hosts permitidos: * (todos)")
    print("\n" + "="*80)
    print("🚀 Iniciando servidor...\n")
    
    try:
        # Iniciar servidor
        proceso = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Mostrar logs
        servidor_listo = False
        for linea in proceso.stdout:
            print(linea, end='')
            
            # Detectar cuando está listo
            if not servidor_listo and ('Listening at:' in linea or 'Booting worker' in linea):
                servidor_listo = True
                print("\n" + "="*80)
                print("✅ SERVIDOR LISTO")
                print("="*80)
                print(f"\n🌐 Accede desde:")
                print(f"   • http://localhost:{PORT}")
                print(f"   • http://127.0.0.1:{PORT}")
                print(f"   • http://<tu-ip>:{PORT}")
                print(f"\n💡 Para detener: Ctrl+C")
                print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("🛑 Deteniendo servidor...")
        print("="*80)
        proceso.terminate()
        try:
            proceso.wait(timeout=5)
            print("✅ Servidor detenido correctamente\n")
        except subprocess.TimeoutExpired:
            proceso.kill()
            print("⚠️  Servidor forzado a detenerse\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
