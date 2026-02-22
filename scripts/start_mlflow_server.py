"""
Script para iniciar el servidor MLflow con CORS habilitado
"""

import subprocess
import sys
import time
import socket
from pathlib import Path
# Variables de entorno para habilitar CORS y configurar hosts permitidos
import os
env = os.environ.copy()

def puerto_disponible(host, port):
    """Verifica si un puerto está disponible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0
    except:
        return False


def iniciar_servidor_mlflow(
    host='0.0.0.0',
    port=8050,
    backend_store=None,
    artifact_root=None,
    allowed_hosts=None
):
    """
    Inicia el servidor MLflow con CORS habilitado
    
    Args:
        host: Host donde escuchará el servidor (0.0.0.0 permite acceso externo)
        port: Puerto del servidor
        backend_store: Ruta para almacenar metadatos (por defecto: ./outputs/hiatal_mlflow/mlruns)
        artifact_root: Ruta para almacenar artifacts (por defecto: ./outputs/hiatal_mlflow/mlartifacts)
        allowed_hosts: Lista de hosts permitidos (por defecto: localhost, 127.0.0.1, y todas las IPs)
    """
    
    # Configurar rutas por defecto
    if backend_store is None:
        raiz = Path.cwd().resolve()
        if raiz.name == 'scripts':
            raiz = raiz.parent
            
        # backend_store = str(raiz / 'outputs' / 'hiatal_mlflow' / 'mlruns')
        backend_store = str(raiz / 'sqlite:///mlflow.db')        
        artifact_root = str(raiz / 'outputs' / 'hiatal_mlflow' / 'mlartifacts')
    
    # Crear directorios si no existen
    Path(backend_store).mkdir(parents=True, exist_ok=True)
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    
    # Verificar si el puerto está disponible
    if not puerto_disponible('localhost', port):
        print(f"❌ Error: El puerto {port} ya está en uso.")
        print(f"   Por favor, detén el proceso que está usando ese puerto o usa otro puerto.")
        return False
    
    # Configurar hosts permitidos
    if allowed_hosts is None:
        print(f"Variables de entorno en None")
        # Por defecto, permitir localhost, 127.0.0.1 y cualquier IP con el puerto
        allowed_hosts = [
            f'localhost:{port}',
            f'127.0.0.1:{port}',            
            '35.173.82.156:8050'
        ]
    
    allowed_hosts_str = ','.join(allowed_hosts)
    
    print("="*80)
    print("INICIANDO SERVIDOR MLFLOW CON CORS")
    print("="*80)
    print(f"\n📋 Configuración:")
    print(f"   Host: {host}")
    print(f"   Puerto: {port}")
    print(f"   Backend Store: {backend_store}")
    print(f"   Artifact Root: {artifact_root}")
    print(f"   CORS: Habilitado para todos los orígenes")
    print(f"   Hosts permitidos: {allowed_hosts_str}")
    
    # Comando para iniciar MLflow con --app-name basic-auth deshabilitado
    # para evitar problemas de validación de host
    comando = [
        'mlflow', 'server',
        '--host', host,
        '--port', str(port),
        # '--backend-store-uri', backend_store,
        '--default-artifact-root', artifact_root,
        '--allowed-hosts', f'{allowed_hosts[0]}, {allowed_hosts[1]}, {allowed_hosts[2]}',
        '--cors-allowed-origins', "*",
        '--serve-artifacts',
    ]  
    
    
    # CORS
    env['MLFLOW_ENABLE_CORS'] = 'true'
    env['MLFLOW_CORS_ALLOW_ORIGIN'] = '*'
    env['MLFLOW_CORS_ALLOW_METHODS'] = 'GET,POST,PUT,DELETE,OPTIONS'
    env['MLFLOW_CORS_ALLOW_HEADERS'] = 'Content-Type,Authorization,X-Requested-With'
    env['MLFLOW_CORS_ALLOW_CREDENTIALS'] = 'true'
    
    # Hosts permitidos - CRÍTICO para solucionar el error
    env['MLFLOW_ALLOWED_HOSTS'] = allowed_hosts_str
    
    # Deshabilitar la validación estricta de host (para desarrollo)
    env['MLFLOW_DISABLE_HOST_VALIDATION'] = 'true'
    
    print(f"\n� Variables de entorno configuradas:")
    print(f"   MLFLOW_ENABLE_CORS: {env.get('MLFLOW_ENABLE_CORS')}")
    print(f"   MLFLOW_CORS_ALLOW_ORIGIN: {env.get('MLFLOW_CORS_ALLOW_ORIGIN')}")
    print(f"   MLFLOW_ALLOWED_HOSTS: {env.get('MLFLOW_ALLOWED_HOSTS')}")
    print(f"   MLFLOW_DISABLE_HOST_VALIDATION: {env.get('MLFLOW_DISABLE_HOST_VALIDATION')}")
    
    print(f"\n🚀 Iniciando servidor...")
    print(f"   Comando: {' '.join(comando)}")
    
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
        
        # Esperar un momento para que el servidor inicie
        print(f"\n⏳ Esperando a que el servidor inicie...")
        time.sleep(3)
        
        # Verificar que el servidor está corriendo
        if proceso.poll() is None:
            print(f"\n✅ Servidor MLflow iniciado correctamente!")
            print(f"\n📊 Accede a la UI en:")
            print(f"   http://localhost:{port}")
            if host == '0.0.0.0':
                print(f"   http://127.0.0.1:{port}")
                print(f"   http://<tu-ip>:{port}")
            
            print(f"\n🌐 CORS configurado:")
            print(f"   - Allow Origin: * (todos los orígenes)")
            print(f"   - Allow Methods: GET, POST, PUT, DELETE, OPTIONS")
            print(f"   - Allow Headers: Content-Type, Authorization, X-Requested-With")
            
            print(f"\n💡 Para detener el servidor, presiona Ctrl+C")
            print("="*80)
            
            # Mantener el proceso corriendo y mostrar logs
            try:
                for linea in proceso.stdout:
                    print(linea, end='')
            except KeyboardInterrupt:
                print(f"\n\n🛑 Deteniendo servidor...")
                proceso.terminate()
                try:
                    proceso.wait(timeout=5)
                    print("✅ Servidor detenido correctamente")
                except subprocess.TimeoutExpired:
                    proceso.kill()
                    print("⚠️  Servidor forzado a detenerse")
            
            return True
        else:
            stdout, stderr = proceso.communicate()
            print(f"\n❌ Error al iniciar el servidor:")
            print(stdout)
            if stderr:
                print(stderr)
            return False
            
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró el comando 'mlflow'")
        print(f"   Asegúrate de tener MLflow instalado:")
        print(f"   pip install mlflow")
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return False


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Inicia el servidor MLflow con CORS habilitado'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host donde escuchará el servidor (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Puerto del servidor (default: 8050)'
    )
    parser.add_argument(
        '--backend-store',
        default=None,
        help='Ruta para almacenar metadatos'
    )
    parser.add_argument(
        '--artifact-root',
        default=None,
        help='Ruta para almacenar artifacts'
    )
    parser.add_argument(
        '--allowed-hosts',
        default=None,
        help='Lista de hosts permitidos separados por coma (ej: localhost,127.0.0.1,35.173.82.156:8050)'
    )
    
    args = parser.parse_args()
    
    # Procesar allowed_hosts
    allowed_hosts = None
    if args.allowed_hosts:
        allowed_hosts = [h.strip() for h in args.allowed_hosts.split(',')]
    
    iniciar_servidor_mlflow(
        host=args.host,
        port=args.port,
        backend_store=args.backend_store,
        artifact_root=args.artifact_root,
        allowed_hosts=allowed_hosts
    )


if __name__ == '__main__':
    main()
