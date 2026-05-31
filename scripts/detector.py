import cv2
from ultralytics import YOLO

# Carga el modelo
modelo = YOLO('yolov8n.pt')

# Abre la camara
camara = cv2.VideoCapture(0)

print("Camara iniciada. Presiona Q para salir.")

while True:
    ret, frame = camara.read()
    if not ret:
        break

    # Detecta personas
    resultados = modelo(frame, classes=[0], conf=0.5, verbose=False)

    personas = []

    for resultado in resultados:
        for caja in resultado.boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            
            # Calcula el centro de cada persona
            centro_x = (x1 + x2) // 2
            centro_y = (y1 + y2) // 2
            personas.append((centro_x, centro_y))

            # Dibuja la caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibuja el centro
            cv2.circle(frame, (centro_x, centro_y), 5, (0, 0, 255), -1)

    # Si hay dos personas mide la distancia entre ellas
    if len(personas) >= 2:
        p1 = personas[0]
        p2 = personas[1]

        # Calcula distancia
        distancia = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5

        # Dibuja linea entre las dos personas
        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # Muestra la distancia en pantalla
        cv2.putText(frame, f"Distancia: {int(distancia)}px", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 0), 2)

        # Alerta si estan muy cerca
        if distancia < 150:
            cv2.putText(frame, "ALERTA: Personas muy cercanas!", 
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)

    # Muestra cuantas personas detecta
    cv2.putText(frame, f"Personas: {len(personas)}", 
                (30, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (255, 255, 255), 2)

    cv2.imshow("Legion IA - Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()