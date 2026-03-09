# HernIA – Aplicación Web para Detección de Hernias Hiatales

HernIA es una aplicación web local de tipo **prototipo** diseñada para la detección automática de **hernias hiatales** a partir de imágenes radiológicas frontales mediante un modelo de **Deep Learning basado en redes neuronales convolucionales (CNN)**.

La aplicación está construida con una arquitectura **desacoplada (frontend–backend)** que separa la interfaz de usuario del motor de inferencia, permitiendo una futura escalabilidad hacia un entorno clínico o de producción.

---

# Descripción General

El sistema está compuesto por dos componentes principales:

## Frontend – Flask

Interfaz web responsable de la interacción con el usuario.

Funciones principales:

- Carga de imágenes radiológicas
- Visualización inmediata de la imagen seleccionada (preview automática)
- Introducción de información del paciente:
  - Nombre del paciente
  - ID del paciente
- Visualización del resultado del modelo
- Visualización de la probabilidad de detección
- Exportación de resultados en **PDF**
- Reinicio del análisis mediante el botón **Nuevo análisis**

---

## Backend – FastAPI

API encargada de ejecutar la inferencia del modelo de Machine Learning.

Funciones principales:

- Recepción de imágenes desde el frontend
- Preprocesamiento de la imagen
- Ejecución del modelo de clasificación
- Retorno del resultado en formato **JSON**

---

# Modelo de Inteligencia Artificial

### Características del modelo

- Tipo: Clasificación binaria
- Entrada: Imagen radiológica frontal
- Salida del modelo:
  - **Probabilidad de hernia hiatal**
  - Clasificación final (hernia / no hernia)

### Flujo de inferencia

1. Recepción de imagen radiológica
2. Preprocesamiento de la imagen
3. Ejecución del modelo 
4. Obtención de probabilidad
5. Interpretación del resultado

---

# Arquitectura del Sistema

Navegador Web
↓
Frontend Flask (http://localhost:5000
)
↓ HTTP POST (imagen)
Backend FastAPI (http://localhost:8000
)
↓
Modelo de Inferencia
↓
Resultado JSON

## Estructura de la APP

hernia_app/
│
├── backend/
│ ├── main.py # API FastAPI
│ └── model.py # Carga e inferencia del modelo
│ ├── model/
│ │ └── production_bundle.pt #Modelo entrenado
│
├── frontend/
│ ├── app.py # Aplicación Flask
│
│ ├── templates/
│ │ └── index.html # Interfaz principal
│
│ └── static/
│ ├── style.css # Estilos de la aplicación
│ └── uploads/ # Imágenes cargadas por usuarios
│
├── venv/ # Entorno virtual de Python
│
├── requirements.txt # Dependencias del proyecto
│
└── README.md

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

2. Introducir datos del paciente

3. Cargue una imagen médica en formato PNG o JPG

4. Presione el botón de análisis

5. La aplicación:

    Envía la imagen al backend

    Ejecuta la inferencia

    Muestra el resultado y la probabilidad de Hernia

5. Utilice la opción Exportar PDF para descargar un reporte con:

    Datos del paciente

    Imagen analizada

    Resultado del diagnóstico

    Probabilidad de Hernia

## Exportación a PDF

La generación del PDF se realiza en el frontend mediante la librería reportlab

El archivo se genera en memoria, sin crear archivos temporales

El PDF incluye datos del paciente, imagen y resultados de inferencia

Nota: la implementación actual está pensada para uso local y de un solo usuario.