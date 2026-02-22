# MLflow con CORS - Guía de Uso

Este documento explica cómo usar MLflow con CORS habilitado para el proyecto de detección de hernia hiatal.

## 📋 Requisitos

```bash
pip install mlflow torch torchvision scikit-learn pandas numpy pillow matplotlib
```

## 🚀 Opción 1: Iniciar servidor MLflow manualmente

### Paso 1: Iniciar el servidor con CORS

```bash
# Desde la raíz del proyecto
python scripts/start_mlflow_server.py
```

O con opciones personalizadas:

```bash
python scripts/start_mlflow_server.py --host 0.0.0.0 --port 8050
```

### Paso 2: Ejecutar el entrenamiento

En otra terminal:

```bash
python scripts/train_model_mlflow.py
```

## 🔄 Opción 2: Inicio automático del servidor

Edita `train_model_mlflow.py` y cambia:

```python
'MLFLOW_AUTO_START': True,  # Cambiar de False a True
```

Luego ejecuta:

```bash
python scripts/train_model_mlflow.py
```

El servidor se iniciará automáticamente y se detendrá al finalizar el entrenamiento.

## 🌐 Configuración de CORS

El servidor MLflow está configurado con las siguientes opciones de CORS:

- **Allow Origin**: `*` (permite todos los orígenes)
- **Allow Methods**: `GET, POST, PUT, DELETE, OPTIONS`
- **Allow Headers**: `Content-Type, Authorization, X-Requested-With`
- **Allow Credentials**: `true`

### Personalizar CORS

Si necesitas restringir los orígenes permitidos, edita `start_mlflow_server.py`:

```python
env = {
    **subprocess.os.environ,
    'MLFLOW_ENABLE_CORS': 'true',
    'MLFLOW_CORS_ALLOW_ORIGIN': 'http://localhost:3000,http://localhost:8080',  # Orígenes específicos
    'MLFLOW_CORS_ALLOW_METHODS': 'GET,POST,PUT,DELETE,OPTIONS',
    'MLFLOW_CORS_ALLOW_HEADERS': 'Content-Type,Authorization',
}
```

## 📊 Acceder a la UI de MLflow

Una vez iniciado el servidor, accede a:

- **Local**: http://localhost:8050
- **Red local**: http://<tu-ip>:8050

## 🔧 Configuración del proyecto

### Estructura de directorios

```
proyecto/
├── data/
│   └── images/
│       ├── normal/
│       └── hernia/
├── outputs/
│   └── hiatal_mlflow/
│       ├── mlruns/        # Metadatos de experimentos
│       └── mlartifacts/   # Artifacts (modelos, gráficos)
└── scripts/
    ├── train_model_mlflow.py
    └── start_mlflow_server.py
```

### Parámetros configurables

En `train_model_mlflow.py`, función `configurar_experimento()`:

```python
config = {
    # MLflow
    'USAR_MLFLOW': True,
    'MLFLOW_EXPERIMENTO': 'hernia-hiatal-production',
    'MLFLOW_TRACKING_URI': 'http://localhost:8050',
    'MLFLOW_REGISTRAR_MODELO': True,
    
    # MLflow Server
    'MLFLOW_AUTO_START': False,  # True para inicio automático
    'MLFLOW_HOST': '0.0.0.0',
    'MLFLOW_PORT': 8050,
    
    # Hiperparámetros
    'TAM_IMAGEN': 512,
    'TAM_BATCH': 8,
    'EPOCAS': 30,
    'DROPOUT': 0.25,
    # ... más parámetros
}
```

## 🐛 Solución de problemas

### Error: "Rejected request with invalid Host header"

Si ves este error:
```
WARNING mlflow.server.fastapi_security: Rejected request with invalid Host header: 35.173.82.156:8050
```

**Solución 1: Usar el script actualizado**

El script ya está configurado para permitir todos los hosts. Simplemente reinicia el servidor:

```bash
python scripts/start_mlflow_server.py
```

**Solución 2: Especificar hosts permitidos manualmente**

```bash
python scripts/start_mlflow_server.py --allowed-hosts "localhost,127.0.0.1,35.173.82.156:8050"
```

**Solución 3: Configurar variable de entorno**

```bash
export MLFLOW_ALLOWED_HOSTS="localhost,127.0.0.1,35.173.82.156:8050,*"
mlflow server --host 0.0.0.0 --port 8050 --backend-store-uri ./outputs/hiatal_mlflow/mlruns
```

**Solución 4: Permitir todos los hosts (desarrollo)**

```bash
export MLFLOW_ALLOWED_HOSTS="*"
python scripts/start_mlflow_server.py
```

⚠️ **Nota de seguridad**: Usar `*` permite cualquier host. Solo úsalo en desarrollo. En producción, especifica los hosts exactos.

### Puerto ya en uso

```bash
# Verificar qué proceso está usando el puerto
lsof -i :8050  # Linux/Mac
netstat -ano | findstr :8050  # Windows

# Detener el proceso o usar otro puerto
python scripts/start_mlflow_server.py --port 8051
```

### Error de CORS en el navegador

Si ves errores de CORS en la consola del navegador:

1. Verifica que el servidor se inició con las variables de entorno correctas
2. Asegúrate de que `MLFLOW_ENABLE_CORS=true` está configurado
3. Revisa que el origen de tu aplicación esté permitido

### MLflow no encontrado

```bash
# Instalar MLflow
pip install mlflow

# Verificar instalación
mlflow --version
```

## 📡 Uso desde aplicaciones web

### JavaScript/Fetch

```javascript
// Obtener experimentos
fetch('http://localhost:8050/api/2.0/mlflow/experiments/search', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({})
})
.then(response => response.json())
.then(data => console.log(data));
```

### Python/Requests

```python
import requests

# Obtener experimentos
response = requests.post(
    'http://localhost:8050/api/2.0/mlflow/experiments/search',
    json={}
)
print(response.json())
```

## 📚 Recursos adicionales

- [Documentación oficial de MLflow](https://mlflow.org/docs/latest/index.html)
- [API REST de MLflow](https://mlflow.org/docs/latest/rest-api.html)
- [Tracking API](https://mlflow.org/docs/latest/tracking.html)

## 🔒 Seguridad

⚠️ **Importante**: La configuración actual permite acceso desde cualquier origen (`*`). 

Para producción:
1. Restringe los orígenes permitidos
2. Implementa autenticación
3. Usa HTTPS
4. Configura un firewall

```python
# Ejemplo de configuración más segura
'MLFLOW_CORS_ALLOW_ORIGIN': 'https://tu-dominio.com',
```

## 📝 Logs y debugging

Los logs del servidor se muestran en la terminal. Para guardarlos:

```bash
python scripts/start_mlflow_server.py > mlflow_server.log 2>&1
```

## 🎯 Métricas registradas

El script registra automáticamente:

- **Parámetros**: tam_imagen, tam_batch, epocas, dropout, etc.
- **Métricas**: val_loss, val_auc, val_accuracy, val_f1, val_sensibilidad, val_especificidad
- **Artifacts**: Modelo entrenado (PyTorch)

## 🔄 Actualizar configuración

Para cambiar la configuración sin editar el código:

```bash
# Variables de entorno
export MLFLOW_TRACKING_URI=http://localhost:8050
export MLFLOW_EXPERIMENT_NAME=mi-experimento

python scripts/train_model_mlflow.py
```
