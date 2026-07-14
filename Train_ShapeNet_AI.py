import os
import numpy as np
from PIL import Image
import cv2  

IMG_SIZE = 64
DATASET_PATH = "C:/Users/migue/Downloads/geometric shapes dataset"

# Solo carpetas válidas
clases = [c for c in os.listdir(DATASET_PATH) 
          if os.path.isdir(os.path.join(DATASET_PATH, c))]

X = []
y = []

MAX_IMGS = 10000  # límite por clase

for label, clase in enumerate(clases):
    ruta_clase = os.path.join(DATASET_PATH, clase)
    
    archivos = os.listdir(ruta_clase)
    
    contador = 0
    
    for archivo in archivos:
        if contador >= MAX_IMGS:
            break
        
        ruta_imagen = os.path.join(ruta_clase, archivo)
        
        try:
            img = Image.open(ruta_imagen)
            
            # 1. Escala de grises
            img = img.convert("L")
            
            # 2. Redimensionar
            img = img.resize((IMG_SIZE, IMG_SIZE))
            
            # 3. A numpy
            img_array = np.array(img)

            # 🔥 4. BINARIZACIÓN DINÁMICA (OTSU)
            _, img_array = cv2.threshold(
                img_array, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # 🔥 (opcional pero recomendado) invertir colores
            # img_array = cv2.bitwise_not(img_array)
            
            # 5. Normalizar (0 o 1)
            img_array = img_array / 255.0
            
            # 6. Flatten
            img_array = img_array.flatten()
            
            X.append(img_array)
            y.append(label)
            
            contador += 1
            
        except:
            print(f"Error al procesar: {ruta_imagen}")
            continue

X = np.array(X)
y = np.array(y)

print("Datos cargados:", X.shape)

import tensorflow as tf
from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(64*64,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)

history = model.fit(
    X_train, y_train,
    epochs=17,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, acc = model.evaluate(X_test, y_test)
print("Accuracy final:", acc)

from sklearn.metrics import confusion_matrix
import numpy as np

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)

model.save("C:/Users/migue/Downloads/modelo_figuras.h5")