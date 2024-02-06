# AgePrediction

En este proyecto se busco resolver el problema de predecir la edad mediante cnns.
Se paso a utilizar la red neuronal pre-entrenada VGG16 añadiéndole algunas capas densas para ajustar la predicción.

| mean_squared_error  |mean_squared_error val   | mae   | mae val  |
|---------------------|-------------------------|-------|----------|
|  100.6699           |  88.3658                | 7.5201| 6.6272   |

El desafió del proyecto fue el preprocesamiento que se hizo en las imágenes, se aplicaron diversos filtros para así poder mejorar la predicción.

## Train-Pipeline

El pipeline de entrenamiento es el siguiente:

* Preprocesamiento y entrenamiento:
* Cargamos las imágenes crudas y las leemos con opencv para así detectar rostros y recortar las imágenes.
* Cargamos la data preprocesada y la dividimos entre X_train y X_test.
* Aplicamos data augmentation.
* Entrenamos el modelo.

## Predict-Pipeline

* Predicción:
* Cargamos la imagen a predecir con opencv y pasamos a detectar el rostro para luego recortar la imagen.
* Instanciamos el modelo con los pesos entrenados y pasamos a predecir la imagen recortada.

## Deployment

Para deployar de forma local la app dirigirse al directorio "deployment" y ejecutar app.py (uvicorn app:app --reload).

