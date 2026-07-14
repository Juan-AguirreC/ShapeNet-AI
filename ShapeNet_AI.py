import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelo
model = load_model("C:/Users/migue/Downloads/modelo_figuras.h5")

# ✅ SOLO 3 clases (igual que entrenamiento)
clases = ["circle", "square", "triangle"]

IMG_SIZE = 32

# Iniciar cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # 1. Escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. ROI (recorte central)
    h, w = gray.shape
    size = 200
    x1 = w//2 - size//2
    y1 = h//2 - size//2
    x2 = x1 + size
    y2 = y1 + size
    
    roi = gray[y1:y2, x1:x2]
    
    # 3. Redimensionar
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    # 🔥 4. BINARIZACIÓN (IGUAL QUE ENTRENAMIENTO)
    _, img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 🔥 OPCIONAL: invertir si ves que falla
    # img = cv2.bitwise_not(img)
    
    # 5. Normalizar
    img = img / 255.0
    
    # 6. Flatten
    img = img.flatten().reshape(1, -1)
    
    # 7. Predicción
    pred = model.predict(img, verbose=0)
    clase = np.argmax(pred)
    texto = clases[clase]

    # Dibujar cuadro y texto
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, texto, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Mostrar
    cv2.imshow("Detector de Figuras", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()