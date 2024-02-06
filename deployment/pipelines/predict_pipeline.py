import cv2 as cv
import os
from model.model_v3 import create_model_v3
import numpy as np

# declaramos el modelo
model_v3 = create_model_v3()  # Reemplaza esto con la función que crea tu modelo

# cargamos los pesos que aprendio el modelo cdo fue entrenado
model_v3.load_weights('./model/model_v3_final_weights.h5')


def cargar_foto(img, model):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if img is not None:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5 )
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]

            face = cv.resize(face, (224, 224))
            face = face.astype('float32') / 255.0  # Normalizar la imagen

            # Preprocesar la imagen para el modelo
            face_array = np.expand_dims(face, axis=0)
            print(face_array.shape)  # Añadir una dimensión para el batch
            #face_array = preprocess_input(face_array)  # Preprocesamiento específico de VGG16

            return model.predict(face_array), face
        else:
            return -1

            
def predict_pipeline(img):

    try:
        # cargamos la foto con la cual se hara la prediccion
        edad, face = cargar_foto(img, model_v3)
        edad = np.round(edad[0][0], 0)
        return edad
    
    except:
        print("entra aca")
        return -1
    
