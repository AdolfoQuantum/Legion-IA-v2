from ultralytics import YOLO

# Carga el modelo (se descarga automatico la primera vez)
modelo = YOLO('yolov8n.pt')

# Corre la deteccion con tu camara web
modelo.predict(
    source=0,
    show=True,
    classes=[0],
    conf=0.5
)