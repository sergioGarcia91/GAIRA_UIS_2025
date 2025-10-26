# **GAIRA – propuesta ganadora INNOVATIC-2025 (UIS)**

<img src="https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/4a81918d410b502b79f58060ad5264dd98e5e373/GAIRA_2025_logo.png" alt="Logo GAIRA" width="500"/>

# Tabla de Contenido
- [Autores y colaboradores](#Autores-y-colaboradores)
- [Descripción](#Descripción)
- [Notebooks](#Notebooks)
- [Talleres](#Talleres)
- [Modelos 3D](#Modelos-3D)

---

## Autores y colaboradores
Autores del proyecto GAIRA
- Francisco Alberto Velandia Patiño
- Sergio Andrés García Arias
- Joaquin Andres Valencia Ortiz

Estudiantes UIS que apoyaron la consolidación
- Shaireth Gizell Carvajal Sinuco
- Valentina Galvis Cobo

Profesionales que brindaron apoyo
- Angélica Álvarez Naranjo
- Carlos Alberto Tavera Sanabria
- César Enrique Llerena Betancour

---
## Descripción

Durante la convocatoria interna [INNOVA-TIC 2025](https://convocatorias.uis.edu.co/convocatoria-innova-tic-2025) , surge [GAIRA: Geología + Artificial Intelligence + Realidad Aumentada](https://comunicaciones.uis.edu.co/conozca-los-ganadores-de-la-convocatoria-innova-tic-2025-y-eventos-tic), cuyo objetivo es:

> **Fortalecer el proceso de enseñanza-aprendizaje de la geología** mediante herramientas digitales basadas en inteligencia artificial, modelación 3D, realidad virtual y aumentada. Utilizadas como apoyo a la divulgación de conceptos geocientíficos complejos, facilitando su comprensión y promoviendo un aprendizaje autónomo, interactivo e inclusivo.

Este repositorio centraliza **notebooks**, **modelos de IA**, **recursos VR**, **material docente** y **scripts** utilizados/desarrollados en GAIRA.

---

## Notebooks
Se incluyen tres notebooks desarrollados en [Google Colaboratory](https://colab.research.google.com).

Su propósito es guiarte en el **entrenamiento, reentrenamiento e inferencia** de modelos **YOLOv8** (framework [Ultralytics](https://docs.ultralytics.com/es/quickstart)) usando Python 3.

> Al hacer clic en el icono ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ubicado al inicio del Notebook, se abrirá el Notebook en Google Colaboratory.

1. [01_GAIRA_Labelme2yolov8](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/81bbca179677fc0b0c9b9cd826b4306f6cd4d8e0/01_GAIRA_Labelme2yolov8.ipynb): Flujo de trabajo para convertir anotaciones de imágenes hechas con [LabelMe](https://github.com/wkentaro/labelme) al formato requerido por [YOLOv8](https://docs.ultralytics.com/es/models/yolov8).
2. [02_GAIRA_Reentrenamiento_YOLOv8](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/81bbca179677fc0b0c9b9cd826b4306f6cd4d8e0/02_GAIRA_Reentrenamiento_YOLOv8.ipynb):  Procedimiento para **entrenar**, **reentrenar** y **reanudar** un modelo YOLOv8 (fine-tuning y manejo de checkpoints).
3. [03_GAIRA_Predecir_YOLOv8](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/81bbca179677fc0b0c9b9cd826b4306f6cd4d8e0/03_GAIRA_Predecir_YOLOv8.ipynb): Inferencia para **clasificar imágenes** en **roca ígnea, sedimentaria o metamórfica** usando diferentes [modelos reentrenados](https://github.com/sergioGarcia91/GAIRA_UIS_2025/tree/81bbca179677fc0b0c9b9cd826b4306f6cd4d8e0/Modelos) de YOLOv8.

---

## Talleres

Se incluyen dos talleres introductorios (formato PDF) para complementar los notebooks y actividades prácticas.

1. [Taller_GAIRA_Modelos_3D](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/44e19f7e679fb271faf16303fe8c8ed57d226aa0/Taller_GAIRA_Modelos_3D.pdf): Introducción a MDT/MDE, carga y visualización en QGIS, generación de bloques-diagrama y flujo básico para publicar modelos 3D (Sketchfab). Incluye preguntas guía y ejercicios con datos de elevación.

2. [Taller_GAIRA_YOLOv8](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/44e19f7e679fb271faf16303fe8c8ed57d226aa0/Taller_GAIRA_YOLOv8.pdf): Análisis comparativo de modelos YOLOv8 (datasets, predicciones, limitaciones y mejoras), trabajo por equipos y entrega en PPT. Propone actividades de inferencia y discusión crítica sobre el uso de IA en geociencias.

---

## Modelos 3D

---


