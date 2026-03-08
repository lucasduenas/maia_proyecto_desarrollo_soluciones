# HiatAI Scan - Manual de Uso

## 1. Introducción

**HiatAI Scan** es un sistema de clasificación automática de radiografías de tórax basado en inteligencia artificial. Utiliza modelos de deep learning (EfficientNet-B3 y DenseNet-121) para detectar y clasificar hernias hiatales, apoyando al personal médico en el proceso de diagnóstico radiológico. 

## 2. Acceso al Sistema

Abra su navegador web y visite `http://localhost:5000` (si lo corre de forma local) o la dirección IP pública del servidor si está desplegado en AWS EC2. La página principal mostrará la interfaz **"HiatAI Scan — Hernia Hiatal Classifier"** junto con el área de carga de imágenes dividida en dos paneles (Adjuntar Radiografía y Resultados).

## 3. Preparación de Imágenes

El sistema acepta radiografías de tórax con las siguientes especificaciones:

- **Formatos soportados:** PNG y JPG/JPEG.
- **Tamaño máximo recomendado:** 10 MB por imagen.
- **Orientación:** Vista frontal (posteroanterior) del tórax.
- **Calidad:** Imágenes nítidas y bien expuestas. El sistema ajusta automáticamente el tamaño pero se recomienda mantener la resolución original para una mayor precisión diagnóstica.

## 4. Carga y Análisis de Radiografías

1. En la página principal, dentro de la sección "Adjuntar Radiografía", haga clic en **"Seleccionar archivo"** (o arrastre la imagen).
2. Seleccione la radiografía de tórax local que desea analizar. 
3. Haga clic en el botón **"Iniciar Análisis"** para enviar la imagen al servidor.
4. Espere unos segundos mientras el modelo deep learning procesa la imagen en el backend.

## 5. Interpretación de Resultados

Una vez procesado, los resultados se muestran en el panel derecho:

- **Clase Predicha:** Diagnóstico principal. Puede ser **Hernia Hiatal Presente** (se detectaron indicadores en rojo) o **Normal** (sin hallazgos de hernia, en verde).
- **Nivel de Confianza (Distribución):** Porcentajes de confianza representados en una barra de progreso. Indica qué tan seguro está el modelo de su predicción de acuerdo a las características aprendidas en entrenamiento.

## 6. Acciones Disponibles

- **Exportar Reporte (PDF):** Genera y descarga un documento formal con la imagen original escaneada, el resultado del diagnóstico y el nivel de confianza.
- **Nuevo Análisis:** Puede cargar una nueva radiografía directamente sobre la misma ventana. Tenga en cuenta que cada nueva carga reemplazará los resultados anteriores en la pantalla.

## ⚠️ 7. Consideraciones Importantes

> **ADVERTENCIA CLÍNICA:** HiatAI Scan es una herramienta de **apoyo diagnóstico** y **NO reemplaza el criterio médico profesional**. Los resultados deben ser siempre revisados e interpretados por personal médico calificado. El sistema proporciona probabilidades basadas en patrones aprendidos de datos históricos y no constituye un diagnóstico médico definitivo.

La precisión del sistema depende directamente de la calidad de la imagen. Imágenes borrosas, mal orientadas o con características no esperadas pueden arrojar resultados atípicos.

## 8. Solución de Problemas de Uso

- **La imagen no carga:** Verifique que el archivo esté en formato PNG o JPG y no supere el límite de tamaño.
- **El procesamiento demora:** El análisis de imágenes médicas a través de DenseNet/EfficientNet es computacionalmente intensivo; espere hasta 30 segundos.
- **Error en el servidor:** Compruebe que el Backend siga ejecutándose en la Terminal 1 (puerto 8000). Refresque la página del Frontend e intente nuevamente.
