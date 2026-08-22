# **Aula RIFT – propuesta ganadora INNOVA-TIC 2026 (UIS)**

Tabla de Contenido
- :busts_in_silhouette: [Autores y colaboradores](#autores_colaboradores)
- :scroll: [Descripción](#descripcion)
- :desktop_computer: [App](#app)
- :man_teacher: [Talleres](#talleres)
- :floppy_disk: [Manuales](#manuales)
- :mirror_ball: [Modelos 3D](#modelos_3d)
- :pray: [Agradecimientos](#agradecimientos)

---
<a id="autores_colaboradores"></a>
## :busts_in_silhouette: Autores y colaboradores
Autores del proyecto GAIRA
- Angélica Alvarez Naranjo
- Sergio Andrés García Arias

Estudiantes UIS que apoyaron la consolidación
- Juan Pablo García Barriendos
- Liseth Yaneth Leal Lizcano


Profesionales que brindaron apoyo
- César Enrique Llerena Betancour

---
<a id="descripcion"></a>
## :scroll: Descripción

Durante la convocatoria interna INNOVA-TIC 2026 se plantea la propuesta: *"Aula RIFT: Resources for Innovation in Field Teaching"*, cuyo objetivo es:

> Implementar un espacio de innovación educativa que integre recursos digitales, modelos geocientíficos tridimensionales y
herramientas de inteligencia artificial para enriquecer la enseñanza de las geociencias y fortalecer el aprendizaje práctico.

---
<a id="app"></a>
## :desktop_computer: App
### Requerimientos
El aplicativo fue desarrollado en `Python==3.10.20`.
```
ultralytics>=8.3.0
PySide6==6.6.3.1
opencv-python==4.10.0.84
numpy==1.26.4
```

### Ejecución
```bash
python -m PyInstaller --noconfirm --clean --onedir --name YOLO_MultiModel_Predictor --runtime-hook rthook_torch_first.py --collect-all ultralytics --collect-submodules ultralytics --collect-binaries torch --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 main.py
```

---
<a id="talleres"></a>
## :man_teacher: Talleres


---
<a id="manuales"></a>
## :floppy_disk: Manuales


---

<a id="modelos_3d"></a>
## :mirror_ball: Modelos 3D



---

<a id="agradecimientos"></a>
## :pray: Agradecimientos

Agradecemos a la **Universidad Industrial de Santander (UIS)**, a la **Vicerrectoría Académica**, a la **Escuela de Geología** y a todas las personas que han hecho posible **Aula RIFT**. En especial, gracias por los estímulos otorgados en el marco del **Portafolio TIC 2026**, [mencionados en las notas oficiales](https://convocatorias.uis.edu.co/convocatoria-innova-tic-2026/index.html).

Invitamos a **docentes y grupos de la UIS** a participar en estas convocatorias y a **sumarse** al desarrollo de propuestas que fortalezcan la docencia, la investigación y la innovación en beneficio del territorio. ¡Construyamos juntos más proyectos de alto impacto! :rocket::handshake::seedling::chart_with_upwards_trend:


