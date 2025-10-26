# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 08:09:08 2024

@author: Sergio Andrés García-Arias
"""


# Importar librerias
from ultralytics import YOLO
import cv2

# Leer el modelo a utilizar
#model = YOLO('rocas_k-x.pt')
#model = YOLO('rocas_ku-x.pt')
#model = YOLO('rocas_kuu-x.pt')
#model = YOLO('rocas_kuuk-x.pt')
#model = YOLO('rocas_kuuku-x.pt') 
#model = YOLO('yolov8x-seg.pt') # probar la de YOLOv8

#model = YOLO('rocas_k-n.pt')
#model = YOLO('rocas_kk-n.pt')
#model = YOLO('rocas_kku-n.pt')
#model = YOLO('rocas_kkuu-n.pt')
#model = YOLO('rocas_kkuuk-n.pt')
#model = YOLO('rocas_kkuuku-n.pt')
#model = YOLO('yolov8n-seg.pt') # probar la de YOLOv8

#model = YOLO('rocas_GAIRA_01-n.pt')
#model = YOLO('rocas_GAIRA_02-n.pt')
#model = YOLO('rocas_GAIRA_03-n.pt')
#model = YOLO('rocas_GAIRA_04-n.pt')
model = YOLO('rocas_GAIRA_05-n.pt')


# Capturar la imagen de la camara 
capCamara = cv2.VideoCapture(0) # 1 para Irium y 0 para la de PC

# Bucle
while True:
    # Leer los fotogramas
    ret, frame = capCamara.read()
    
    # Predecir resultado
    resultadosCNN = model.predict(frame, 
                                  imgsz=640, 
                                  conf=0.3,
                                  stream_buffer=True)
    
    # Mostrar resultados
    resultadoFrame = resultadosCNN[0].plot()
    
    # Mostrar captura
    cv2.imshow('Tipo de Roca', resultadoFrame) # los resultados de la prediccion

        
    # Cerrar la ventana al pulsar la tecla Esc
    if cv2.waitKey(1) == 27: 
        break

capCamara.release()
cv2.destroyAllWindows()