# **GAIRA – propuesta ganadora INNOVA-TIC 2025 (UIS)**

<p align="center">
<img src="https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/4a81918d410b502b79f58060ad5264dd98e5e373/GAIRA_2025_logo.png" alt="Logo GAIRA" width="300"/>
</p>

# Tabla de Contenido
- [Autores y colaboradores](#Autores-y-colaboradores)
- [Descripción](#Descripción)
- [Notebooks](#Notebooks)
- [Talleres](#Talleres)
- [Modelos 3D](#Modelos-3D)
- [:pray: Agradecimientos](Agradecimientos)

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

Para potenciar GAIRA en el marco del Portafolio TIC 2025 de la UIS, la Vicerrectoría Académica entregó como [estímulo un visor Oculus Meta Quest 3](https://comunicaciones.uis.edu.co/uis-realizo-entrega-de-estimulos-dentro-de-la-convocatoria-ova-e-innova-tic-2025). Este equipo habilita la visualización inmersiva de los gemelos digitales publicados en Sketchfab, permitiendo que los estudiantes simulen trabajo de campo y clasificación de muestras en realidad virtual, mejorando la comprensión sin necesidad de desplazamientos.

Este repositorio centraliza **notebooks**, **modelos de IA**, **recursos VR**, **material docente** y **scripts** utilizados/desarrollados en GAIRA.

<p align="center">
<img src="https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/a70558fd0d4194bbaeab8daab789410d7b4d40a7/Innovatic-eventos-TIC-ganadores-2025.jpeg" alt="Ganadores INNOVATIC 2025 UIS" width="300"/>
<img src="https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/a70558fd0d4194bbaeab8daab789410d7b4d40a7/Geologia-profesores.jpeg" alt="Profesores GAIRA" width="300"/>
<img src="https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/fb0ff217a742004930f6d6e84a66a65b7857f7ef/Entrega-de-equipos-Gaira.jpg" alt="Gafas GAIRA" width="300"/>
</p>

---

## Notebooks
Se incluyen tres notebooks desarrollados en [Google Colaboratory](https://colab.research.google.com).

Su propósito es guiarte en el **entrenamiento, reentrenamiento e inferencia** de modelos **YOLOv8** (framework [Ultralytics](https://docs.ultralytics.com/es/quickstart)) usando Python 3.

> Al hacer clic en el icono ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ubicado al inicio del Notebook, se abrirá el Notebook en Google Colaboratory.

1. [01_GAIRA_Labelme2yolov8](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/81bbca179677fc0b0c9b9cd826b4306f6cd4d8e0/01_GAIRA_Labelme2yolov8.ipynb): Flujo de trabajo para convertir anotaciones de imágenes hechas con [LabelMe](https://github.com/wkentaro/labelme) al formato requerido por [YOLOv8](https://docs.ultralytics.com/es/models/yolov8).
2. [02_GAIRA_Reentrenamiento_YOLOv8](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/81bbca179677fc0b0c9b9cd826b4306f6cd4d8e0/02_GAIRA_Reentrenamiento_YOLOv8.ipynb):  Procedimiento para **entrenar**, **reentrenar** y **reanudar** un modelo YOLOv8 (fine-tuning y manejo de checkpoints).
3. [03_GAIRA_Predecir_YOLOv8](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/81bbca179677fc0b0c9b9cd826b4306f6cd4d8e0/03_GAIRA_Predecir_YOLOv8.ipynb): Inferencia para **clasificar imágenes** en **roca ígnea, sedimentaria o metamórfica** usando diferentes [modelos reentrenados](https://github.com/sergioGarcia91/GAIRA_UIS_2025/tree/81bbca179677fc0b0c9b9cd826b4306f6cd4d8e0/Modelos) de YOLOv8.

> *Nota (uso local en tiempo real):* El script [`TipoRoca.py`](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/498109fc148b63fd867c3dcb724738deeddc3cb1/TipoRoca.py) permite realizar inferencias en tiempo real con la cámara del PC usando los modelos YOLOv8 reentrenados del proyecto. Este script hace parte del trabajo de [García-Arias & Velandia (2025)](https://doi.org/10.24050/reia.v22i43.1813).

---

## Talleres

Se incluyen dos talleres introductorios (formato PDF) para complementar los notebooks y actividades prácticas.

1. [Taller_GAIRA_Modelos_3D](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/44e19f7e679fb271faf16303fe8c8ed57d226aa0/Taller_GAIRA_Modelos_3D.pdf): Introducción a MDT/MDE, carga y visualización en QGIS, generación de bloques-diagrama y flujo básico para publicar modelos 3D (Sketchfab). Incluye preguntas guía y ejercicios con datos de elevación.

2. [Taller_GAIRA_YOLOv8](https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/44e19f7e679fb271faf16303fe8c8ed57d226aa0/Taller_GAIRA_YOLOv8.pdf): Análisis comparativo de modelos YOLOv8 (datasets, predicciones, limitaciones y mejoras), trabajo por equipos y entrega en PPT. Propone actividades de inferencia y discusión crítica sobre el uso de IA en geociencias.

---

## Modelos 3D

Considerando la disponibilidad de muestras de roca de la [Escuela de Geología](https://geologia.uis.edu.co/eisi) de la UIS, se generaron **gemelos digitales** mediante la técnica de **fotogrametría**. Estos modelos están publicados en el **perfil de Sketchfab** de la Escuela de Geología: https://sketchfab.com/EscgeoUIS, donde se encuentran organizados en las siguientes colecciones:

- [Colección General - Rocas Metamórficas](https://sketchfab.com/EscgeoUIS/collections/coleccion-general-rocas-metamorficas-20f35655e942416f826af26a88c4b33d)
- [Colección General - Rocas Ígneas](https://sketchfab.com/EscgeoUIS/collections/coleccion-general-rocas-igneas-041eee88801c4adeaca944a83981fe10)
- [Colección General - Fósiles](https://sketchfab.com/EscgeoUIS/collections/coleccion-general-fosiles-a9aa24af24734ba0b485a8efb7a07698)
- [Colección General - Rocas Sedimentarias](https://sketchfab.com/EscgeoUIS/collections/coleccion-general-rocas-sedimentarias-ce4bd7733ba649a590e6d0fefc3c9d14)
- [Modelos 3D del terreno](https://sketchfab.com/EscgeoUIS/collections/modelos-3d-del-terreno-8e746843c16a4a2b9f2044974f291963)
- [Modelos Digitales de Afloramiento](https://sketchfab.com/EscgeoUIS/collections/modelos-digitales-de-afloramiento-f4e4f6dfc968476ea4bf07b2b722acbd)

<p align="center">
<img src="https://github.com/sergioGarcia91/GAIRA_UIS_2025/blob/a70558fd0d4194bbaeab8daab789410d7b4d40a7/Sketchfab_Escuela_de_Geologia_UIS.png" alt="Perfil Sketchfab Escuela de Geología UIS" width="700"/>
</p>

---

## :pray: Agradecimientos

Agradecemos a la **Universidad Industrial de Santander (UIS)**, a la **Vicerrectoría Académica**, a la **Escuela de Geología** y a todas las personas que han hecho posible **GAIRA**. En especial, gracias por los estímulos otorgados en el marco del **Portafolio TIC 2025**, mencionados en las notas oficiales: [entrega de estímulos OVA e INNOVA-TIC 2025](https://comunicaciones.uis.edu.co/uis-realizo-entrega-de-estimulos-dentro-de-la-convocatoria-ova-e-innova-tic-2025) y [ganadores INNOVA-TIC 2025 y Eventos TIC](https://comunicaciones.uis.edu.co/conozca-los-ganadores-de-la-convocatoria-innova-tic-2025-y-eventos-tic).

Invitamos a **docentes y grupos de la UIS** a participar en estas convocatorias y a **sumarse** al desarrollo de propuestas que fortalezcan la docencia, la investigación y la innovación en beneficio del territorio. ¡Construyamos juntos más proyectos de alto impacto! :rocket::handshake::seedling::chart_with_upwards_trend:



