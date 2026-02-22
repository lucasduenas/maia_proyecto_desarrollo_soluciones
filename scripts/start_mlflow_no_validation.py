"""
Script alternativo para iniciar MLflow SIN validación de host
Útil cuando tienes problemas con el error "Invalid Host header"
"""

import subprocess
import sys
import os
from pathlib import Path


def iniciar_mlflow_sin_validacion(port=8050):
    """
    Inicia MLflow con validación de host deshabilitada
    """
    
    # Configurar rutas
    raiz = Path.cwd().resolve()
    if raiz.name == 'scripts':
        raiz = raiz.parent
    
    backend_store = str(raiz / 'outputs' / 'hiatal_mlflow' / 'mlruns')
    artifact_root = str(raiz / 'outputs' / 'hiatal_mlflow' / 'mlartifacts')
    
    # Crear directorios
    Path(backend_store).mkdir(parents=True, exist_ok=True)
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("INICIANDO MLFLOW SIN VALIDACIÓN DE HOST")
    print("="*80)
    print("\n⚠️  ADVERTENCIA: Este modo deshabilita la validación de host")
    print("   Solo usar en desarrollo/testing")
    print("\n📋 Configuración:")
    print(f"   Puerto: {port}")
    print(f"   Backend: {backend_store}")
    print(f"   Artifacts: {artifact_root}")
    
    # Configurar TODAS las variables de entorno necesarias
    env = os.environ.copy()
    
    # CORS - Permitir todo
    env['MLFLOW_ENABLE_CORS'] = 'true'
    env['MLFLOW_CORS_ALLOW_ORIGIN'] = '*'
    env['MLFLOW_CORS_ALLOW_METHODS'] = 'GET,POST,PUT,DELETE,OPTIONS,PATCH,HEAD'
    env['MLFLOW_CORS_ALLOW_HEADERS'] = '*'
    env['MLFLOW_CORS_ALLOW_CREDENTIALS'] = 'true'
    env['MLFLOW_CORS_EXPOSE_HEADERS'] = '*'
    env['MLFLOW_CORS_MAX_AGE'] = '3600'
    
    # Hosts - Permitir TODOS
    env['MLFLOW_ALLOWED_HOSTS'] = '*'
    
    # Deshabilitar validaciones de seguridad (SOLO PARA DESARROLLO)
    env['MLFLOW_DISABLE_HOST_VALIDATION'] = 'true'
    env['MLFLOW_DISABLE_CSRF_PROTECTION'] = 'true'
    
    # Configuración adicional
    env['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
    
    print("\n🔧 Variables de entorno:")
    for key in sorted(env.keys()):
        if key.startswith('MLFLOW_'):
            print(f"   {key}: {env[key]}")
    
    # Comando MLflow
    comando = [
        sys.executable, '-m', 'mlflow', 'server',
        '--host', '0.0.0.0',
        '--port', str(port),
        '--backend-store-uri', backend_store,
        '--default-artifact-root', artifact_root,
        '--serve-artifacts',
        '--gunicorn-opts', '--timeout 120 --workers 1',
    ]
    
    print(f"\n🚀 Iniciando servidor...")
    print(f"   Comando: {' '.join(comando)}")
    print("\n" + "="*80)
    
    try:
        # Iniciar el servidor
        proceso = subprocess.Popen(
            comando,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print("\n⏳ Esperando a que el servidor inicie...\n")
        
        # Mostrar logs en tiempo real
        try:
            for linea in proceso.stdout:
                print(linea, end='')
                
                # Detectar cuando el servidor está listo
                if 'Listening at:' in linea or 'Application startup complete' in linea:
                    print("\n" + "="*80)
                    print("✅ SERVIDOR INICIADO CORRECTAMENTE")
                    print("="*80)
                    print(f"\n📊 Accede a la UI en: http://localhost:{port}")
                    print(f"🌐 O desde red: http://35.173.82.156:{port}")
                    print("\n💡 Para detener: Ctrl+C")
                    print("="*80 + "\n")
                    
        except KeyboardInterrupt:
            print(f"\n\n🛑 Deteniendo servidor...")
            proceso.terminate()
            try:
                proceso.wait(timeout=5)
                print("✅ Servidor detenido")
            except subprocess.TimeoutExpired:
                proceso.kill()
                print("⚠️  Servidor forzado a detenerse")
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró MLflow")
        print(f"   Instala con: pip install mlflow")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Inicia MLflow sin validación de host (solo desarrollo)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Puerto del servidor (default: 8050)'
    )
    
    args = parser.parse_args()
    
    iniciar_mlflow_sin_validacion(port=args.port)
