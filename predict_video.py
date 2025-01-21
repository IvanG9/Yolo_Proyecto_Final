import os
from ultralytics import YOLO
import cv2

# Configuración de rutas
VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'video2.mp4')
video_path_out = '{}_out4.mp4'.format(video_path)

# Leer el video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

# Configuración del escritor de video (para guardar el resultado)
out = cv2.VideoWriter(
    video_path_out, 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    int(cap.get(cv2.CAP_PROP_FPS)), 
    (W, H)
)

# Ruta del modelo personalizado
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Cargar el modelo
model = YOLO(model_path)  # Cargar un modelo YOLO personalizado
threshold = 0.5  # Umbral de confianza
class_name_dict = {0: 'phone'}  # Diccionario de nombres de clases

# Procesar el video frame por frame
while ret:
    results = model(frame)[0]  # Obtener resultados del modelo
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            # Dibujar el rectángulo de detección
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            # Poner la etiqueta de la clase detectada
            cv2.putText(
                frame, 
                class_name_dict[int(class_id)].upper(), 
                (int(x1), int(y1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.3, 
                (0, 255, 0), 
                3, 
                cv2.LINE_AA
            )
    out.write(frame)  # Escribir el frame procesado en el archivo de salida
    ret, frame = cap.read()  # Leer el siguiente frame

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
