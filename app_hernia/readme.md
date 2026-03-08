# HernIA – Aplicación Web para Detección de Hernias (Prototipo)

HernIA es una aplicación web local de tipo **prototipo** diseñada para la detección de hernias a partir de imágenes médicas (por ejemplo, radiografías) mediante un modelo de aprendizaje automático.

La aplicación está construida con una arquitectura desacoplada que separa la interfaz de usuario del motor de inferencia, permitiendo una futura escalabilidad hacia un entorno de producción.

---

## Descripción General

El sistema está compuesto por:

- **Frontend** desarrollado en Flask:
  - Carga de imágenes médicas
  - Visualización de la imagen cargada
  - Presentación de resultados de inferencia
  - Exportación del diagnóstico en formato PDF

- **Backend** desarrollado en FastAPI y ejecutado con Uvicorn:
  - Recepción de imágenes
  - Ejecución del modelo de inferencia
  - Retorno de resultados estructurados en formato JSON

Actualmente, el modelo de inferencia es un **placeholder**, preparado para ser reemplazado por un modelo real de Machine Learning o Deep Learning.

---

## Arquitectura del Sistema

Navegador Web
↓
Frontend Flask (http://localhost:5000
)
↓ HTTP POST (imagen)
Backend FastAPI (http://localhost:8000
)
↓
Modelo de Inferencia (placeholder)

## Estructura de la APP

hernia_app/
├── backend/
│ ├── main.py # Aplicación FastAPI
│ └── model.py # Lógica de inferencia (placeholder)
│
├── frontend/
│ ├── app.py # Aplicación Flask
│ ├── templates/
│ │ └── index.html # Interfaz de usuario
│ └── static/
    └── uploads/  # Carpeta para guardar imágenes subidas por los usuarios
│   └── style.css # Estilos CSS
│
├── venv/ # Entorno virtual de Python
└── requirements.txt # Dependencias del proyecto

## Instalación

### 1. Clonar o crear el proyecto

git clone <url-del-repositorio>
cd hernia_app

### 2. Crear y activar un entorno virtual

python -m venv venv
source venv/bin/activate
venv\Scripts\activate

### 3. Instalar dependencias

pip install -r requirements.txt


## Ejecución de la Aplicación

### Terminal 1 – Backend (FastAPI)

source venv/bin/activate
cd backend
uvicorn main:app --reload --port 8000


### Terminal 2 – Frontend (Flask)

source venv/bin/activate
cd frontend
python app.py

## Uso de la Aplicación

1. Acceda desde el navegador a http://localhost:5000

2. Cargue una imagen médica en formato PNG o JPG

3. Presione el botón de análisis

4. La aplicación:

    Envía la imagen al backend

    Ejecuta la inferencia

    Muestra el resultado y la confianza

5. Utilice la opción Exportar PDF para descargar un reporte con:

    Imagen analizada

    Resultado del diagnóstico

    Nivel de confianza

## Exportación a PDF

La generación del PDF se realiza en el frontend mediante la librería reportlab

El archivo se genera en memoria, sin crear archivos temporales

El PDF incluye imagen y resultados de inferencia

Nota: la implementación actual está pensada para uso local y de un solo usuario.