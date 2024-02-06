from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization

def create_model_v3():
    # Cargar la base del modelo VGG16 preentrenado
    model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Congelar los pesos de todas las capas excepto las últimas 4
    for layer in model_vgg16.layers[:-2]:
        layer.trainable = False

    x = Flatten()(model_vgg16.output)

    # Capas densas personalizadas
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Capa de salida para regresión
    predictions = Dense(1)(x)

    # Crear el modelo final
    model = Model(inputs=model_vgg16.input, outputs=predictions)

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

    return model