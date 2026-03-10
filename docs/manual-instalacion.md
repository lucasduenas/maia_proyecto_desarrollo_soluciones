# HiatAI Scan - Manual de Instalación

## 1. Requisitos Previos

El sistema HiatAI Scan requiere el siguiente software instalado en el equipo o servidor:

- **Python 3.10 o superior** con pip.
- **Git** para clonar el repositorio.
- **AWS CLI** configurado con credenciales válidas (para descarga de modelos).
- **Docker** (opcional, para despliegue en contenedor).
- **Hardware mínimo:** 4 GB de RAM disponibles y 3 GB de espacio en disco.
- **Red:** Puertos 5000 (Frontend Flask) y 8000 (Backend FastAPI) disponibles en el host.

---

## Opción A: Despliegue con Entorno Virtual de Python (Desarrollo)

**1. Clonar el repositorio:**
```bash
git clone https://github.com/lucasduenas/maia_proyecto_desarrollo_soluciones.git
cd maia_proyecto_desarrollo_soluciones/app_hernia
```

**2. Configurar credenciales de AWS:**
Para acceder a los datos y modelos almacenados externamente en S3:
```bash
aws configure
```

**3. Crear y activar un entorno virtual de Python:**
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
```

**4. Instalar las dependencias del proyecto:**
```bash
pip install -r requirements.txt
```

**5. Descargar los modelos entrenados desde S3 usando DVC:**
```bash
dvc pull
```

**6. Inicio del Sistema:**
La aplicación requiere dos terminales ejecutándose de forma simultánea. Asegúrese de tener el entorno virtual activado en ambas.

*Terminal 1 – Backend (FastAPI):*
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
*El sistema cargará el modelo de clasificación (EfficientNet-B3 o DenseNet-121) durante el arranque.*

*Terminal 2 – Frontend (Flask):*
```bash
cd frontend
python app.py
```

---

## Opción B: Despliegue con Docker (Producción)

El proyecto incluye dos Dockerfiles para compilar cada servicio en un contenedor, lo que evita conflictos de dependencias en máquinas servidoras como AWS EC2. Ambos contenedores utilizan la imagen base ligera `python:3.11-slim`.

**1. Construir y ejecutar el Backend:**
```bash
docker build -f Dockerfile.backend -t hernia-backend .
docker run --rm -p 8000:8000 hernia-backend
```

**2. Construir y ejecutar el Frontend:**
```bash
docker build -f Dockerfile.frontend -t hernia-frontend .
docker run --rm -p 5000:5000 hernia-frontend
```

---

## 2. Verificación del Sistema

Para confirmar que la instalación fue exitosa y los servicios están correctamente enlazados, abra su navegador web (Google Chrome, Firefox, etc.) e ingrese la siguiente dirección:

**`http://localhost:5000`**

Debería cargar de inmediato la interfaz principal de HiatAI Scan con el título "Clasificación de hernias hiatales". Si recibe un mensaje de "Sitio Inaccesible", revise la consola del Backend y del Frontend en busca de errores y consulte la sección de Solución de Problemas de este manual.

---

## 3. Solución de Problemas
- **Puerto ocupado:** Verifique que el puerto 8000 o 5000 estén libres. En Windows puede usar `netstat -an | findstr 8000`. En linux puede usar `lsof -i :8000`.
- **Módulo no encontrado:** Asegúrese de que el entorno virtual esté activado antes de ejecutar `uvicorn` o `python` si no usa Docker.
- **Error en dvc pull:** Verifique que las credenciales de AWS CLI estén configuradas correctamente y coincidan con los permisos del bucket S3 de MAIA.
- **Modelo no cargado:** Ejecute `dvc pull` nuevamente para descargar el archivo del modelo desde S3 y verifique la variable `MODEL_PATH`.
