# 🔧 Solución al Error "Rejected request with invalid Host header"

## ❌ El Error

```
2026/02/22 01:26:55 WARNING mlflow.server.fastapi_security: Rejected request with invalid Host header: 35.173.82.156:8050
```

Este error ocurre porque MLflow valida el header `Host` de las peticiones HTTP para prevenir ataques de seguridad. Tu servidor está rechazando peticiones desde la IP `35.173.82.156`.

## ✅ Soluciones Rápidas

### Opción 1: Script sin validación (MÁS FÁCIL - Recomendado para desarrollo)

```bash
# Detener el servidor actual (Ctrl+C)

# Usar el script que deshabilita la validación de host
python scripts/start_mlflow_no_validation.py
```

Este script deshabilita completamente la validación de host y es la solución más simple para desarrollo.

⚠️ **Solo para desarrollo**: No uses este script en producción.

### Opción 2: Usar el script Python actualizado

```bash
# Detener el servidor actual (Ctrl+C)

# Iniciar con el script actualizado
python scripts/start_mlflow_server.py
```

El script ya está configurado para permitir todos los hosts automáticamente.

### Opción 2: Usar el script Bash/Batch

**Linux/Mac:**
```bash
chmod +x scripts/start_mlflow_cors.sh
./scripts/start_mlflow_cors.sh 8050 35.173.82.156
```

**Windows:**
```cmd
scripts\start_mlflow_cors.bat 8050 35.173.82.156
```

### Opción 3: Configurar manualmente con variables de entorno

**Linux/Mac:**
```bash
export MLFLOW_ALLOWED_HOSTS="*"
export MLFLOW_ENABLE_CORS="true"
export MLFLOW_CORS_ALLOW_ORIGIN="*"

mlflow server \
    --host 0.0.0.0 \
    --port 8050 \
    --backend-store-uri ./outputs/hiatal_mlflow/mlruns \
    --default-artifact-root ./outputs/hiatal_mlflow/mlartifacts \
    --serve-artifacts
```

**Windows (PowerShell):**
```powershell
$env:MLFLOW_ALLOWED_HOSTS="*"
$env:MLFLOW_ENABLE_CORS="true"
$env:MLFLOW_CORS_ALLOW_ORIGIN="*"

mlflow server `
    --host 0.0.0.0 `
    --port 8050 `
    --backend-store-uri ./outputs/hiatal_mlflow/mlruns `
    --default-artifact-root ./outputs/hiatal_mlflow/mlartifacts `
    --serve-artifacts
```

**Windows (CMD):**
```cmd
set MLFLOW_ALLOWED_HOSTS=*
set MLFLOW_ENABLE_CORS=true
set MLFLOW_CORS_ALLOW_ORIGIN=*

mlflow server --host 0.0.0.0 --port 8050 --backend-store-uri ./outputs/hiatal_mlflow/mlruns --default-artifact-root ./outputs/hiatal_mlflow/mlartifacts --serve-artifacts
```

### Opción 4: Especificar hosts específicos (Más seguro)

Si conoces las IPs exactas que necesitas permitir:

```bash
export MLFLOW_ALLOWED_HOSTS="localhost,127.0.0.1,35.173.82.156:8050,tu-dominio.com:8050"

python scripts/start_mlflow_server.py
```

O con el script:

```bash
python scripts/start_mlflow_server.py --allowed-hosts "localhost,127.0.0.1,35.173.82.156:8050"
```

## 🔍 Verificar que funciona

Después de reiniciar el servidor, deberías ver en los logs:

```
📋 Configuración:
   Host: 0.0.0.0
   Puerto: 8050
   ...
   Hosts permitidos: localhost,127.0.0.1,localhost:8050,127.0.0.1:8050,*
```

Y el error debería desaparecer.

## 🔒 Consideraciones de Seguridad

### Para Desarrollo (Local)
```bash
export MLFLOW_ALLOWED_HOSTS="*"  # ✅ OK - Permite todos los hosts
```

### Para Producción (Servidor público)
```bash
# ❌ NO usar "*" en producción
# ✅ Especificar hosts exactos:
export MLFLOW_ALLOWED_HOSTS="tu-dominio.com,api.tu-dominio.com,35.173.82.156:8050"
```

## 📝 Explicación Técnica

El error ocurre porque:

1. MLflow valida el header `Host` de cada petición HTTP
2. Por defecto, solo permite `localhost` y `127.0.0.1`
3. Cuando accedes desde una IP pública (como `35.173.82.156`), la petición es rechazada
4. La variable `MLFLOW_ALLOWED_HOSTS` controla qué hosts son válidos

## 🆘 Si el problema persiste

1. **Verifica que las variables de entorno están configuradas:**
   ```bash
   echo $MLFLOW_ALLOWED_HOSTS  # Linux/Mac
   echo %MLFLOW_ALLOWED_HOSTS%  # Windows CMD
   $env:MLFLOW_ALLOWED_HOSTS    # Windows PowerShell
   ```

2. **Reinicia completamente el servidor:**
   - Detén el proceso actual (Ctrl+C)
   - Cierra la terminal
   - Abre una nueva terminal
   - Configura las variables de entorno
   - Inicia el servidor nuevamente

3. **Verifica la versión de MLflow:**
   ```bash
   mlflow --version
   ```
   
   Si es muy antigua, actualiza:
   ```bash
   pip install --upgrade mlflow
   ```

4. **Revisa los logs completos:**
   El servidor debería mostrar al inicio qué hosts están permitidos.

## 📚 Referencias

- [MLflow Security Documentation](https://mlflow.org/docs/latest/auth/index.html)
- [FastAPI CORS Documentation](https://fastapi.tiangolo.com/tutorial/cors/)

## 💡 Tip

Para evitar este problema en el futuro, siempre inicia el servidor con:

```bash
python scripts/start_mlflow_server.py
```

Este script ya tiene toda la configuración necesaria.
