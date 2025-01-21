from ultralytics import YOLO

if __name__ == "__main__":
    # Cargar el modelo
    model = YOLO("yolo11n.pt")  # Cargar cualquier tipo de modelo

    # Entrenar el modelo
    results = model.train(
        data="configNUEVO_OBJETO.yaml",
        epochs=100,
        batch=16,
        device=0,  # Usar GPU (0) o CPU (-1)
        workers=0  # Desactivar multiprocesamiento para evitar errores en Windows
    )
