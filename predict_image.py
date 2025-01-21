from ultralytics import YOLO
import numpy as np
import os

# Configurar las rutas
IMAGEN_DIR = os.path.join('.', 'imagenes')
imagen_path = os.path.join(IMAGEN_DIR, 'image4.jpg')
imagen_out_path = '{}_out.jpg'.format(imagen_path)
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Cargar el modelo YOLO preentrenado
model = YOLO(model_path)  # Cargar un modelo personalizado

# Realizar la predicción sobre una imagen
detection_output = model.predict(
    source=imagen_path,  # Ruta de la imagen de entrada
    conf=0.75,           # Confianza mínima
    save=True            # Guardar los resultados en disco
)

# Imprimir la salida del tensor (opcional)
print(detection_output)

# Si se utiliza GPU, transferir los datos a la CPU para procesarlos
if model.device.type == 'cuda':
    print(detection_output[0].cpu().numpy())
else:
    print(detection_output[0].numpy())
