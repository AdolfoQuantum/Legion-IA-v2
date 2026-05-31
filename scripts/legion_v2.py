import cv2
import numpy as np
from ultralytics import YOLO

modelo = YOLO('yolov8n-pose.pt')

def calcular_probabilidad_saludo(dist_cabezas, dist_cuerpos, 
                                  muñecas_arriba):
    """
    Calcula probabilidad de saludo basada en:
    - Distancia entre cabezas
    - Distancia entre cuerpos
    - Si alguna muñeca esta levantada
    """
    probabilidad = 0

    # Si las cabezas estan cerca suma probabilidad
    if dist_cabezas < 100:
        probabilidad += 40
    elif dist_cabezas < 150:
        probabilidad += 20

    # Si los cuerpos estan a distancia media suma probabilidad
    if 100 < dist_cuerpos < 250:
        probabilidad += 30
    elif dist_cuerpos < 100:
        probabilidad -= 20

    # Si hay una muñeca levantada suma probabilidad
    if muñecas_arriba:
        probabilidad += 30

    return max(0, min(100, probabilidad))

def clasificar_accion(persona1_kpts, persona2_kpts, 
                       distancia, escala_x, escala_y):
    try:
        # Cabezas (nariz = punto 0)
        nariz_p1 = persona1_kpts[0][:2] * [escala_x, escala_y]
        nariz_p2 = persona2_kpts[0][:2] * [escala_x, escala_y]
        dist_cabezas = np.linalg.norm(nariz_p1 - nariz_p2)

        # Muñecas y hombros
        muñeca_izq_p1 = persona1_kpts[9][:2]
        muñeca_der_p1 = persona1_kpts[10][:2]
        muñeca_izq_p2 = persona2_kpts[9][:2]
        muñeca_der_p2 = persona2_kpts[10][:2]
        hombro_izq_p1 = persona1_kpts[5][:2]
        hombro_der_p1 = persona1_kpts[6][:2]
        hombro_izq_p2 = persona2_kpts[5][:2]
        hombro_der_p2 = persona2_kpts[6][:2]

        altura_hombros_p1 = (hombro_izq_p1[1] + 
                              hombro_der_p1[1]) / 2
        altura_hombros_p2 = (hombro_izq_p2[1] + 
                              hombro_der_p2[1]) / 2

        dist_muñecas = min(
            np.linalg.norm(muñeca_izq_p1 - muñeca_izq_p2),
            np.linalg.norm(muñeca_izq_p1 - muñeca_der_p2),
            np.linalg.norm(muñeca_der_p1 - muñeca_izq_p2),
            np.linalg.norm(muñeca_der_p1 - muñeca_der_p2)
        )

        altura_muñeca_p1 = (muñeca_izq_p1[1] + 
                             muñeca_der_p1[1]) / 2
        altura_muñeca_p2 = (muñeca_izq_p2[1] + 
                             muñeca_der_p2[1]) / 2

        muñecas_arriba = (altura_muñeca_p1 < altura_hombros_p1 or
                          altura_muñeca_p2 < altura_hombros_p2)

        # Probabilidad de saludo
        prob_saludo = calcular_probabilidad_saludo(
            dist_cabezas, distancia, muñecas_arriba
        )

        # Clasificacion principal
        if distancia > 200:
            return ("DISTANCIA SEGURA", (0, 255, 0),
                    "Sin interaccion", 
                    prob_saludo, dist_cabezas,
                    nariz_p1, nariz_p2)

        elif 100 < distancia <= 200:
            if prob_saludo > 50:
                return ("POSIBLE SALUDO", (0, 255, 150),
                        f"Probabilidad saludo: {prob_saludo}%",
                        prob_saludo, dist_cabezas,
                        nariz_p1, nariz_p2)
            elif dist_muñecas < 100:
                return ("ABRAZO", (0, 165, 255),
                        "Contacto amistoso",
                        prob_saludo, dist_cabezas,
                        nariz_p1, nariz_p2)
            else:
                return ("PROXIMIDAD", (0, 255, 255),
                        "Personas cercanas",
                        prob_saludo, dist_cabezas,
                        nariz_p1, nariz_p2)

        else:
            if muñecas_arriba and dist_muñecas < 80:
                return ("PELIGRO", (0, 0, 255),
                        "Posible agresion!",
                        prob_saludo, dist_cabezas,
                        nariz_p1, nariz_p2)
            elif dist_muñecas < 60:
                return ("ABRAZO", (0, 165, 255),
                        "Abrazo cercano",
                        prob_saludo, dist_cabezas,
                        nariz_p1, nariz_p2)
            else:
                return ("PRECAUCION", (0, 255, 255),
                        "Muy cercanos",
                        prob_saludo, dist_cabezas,
                        nariz_p1, nariz_p2)

    except Exception:
        if distancia < 150:
            return ("PRECAUCION", (0, 255, 255),
                    "Muy cercanos", 0, 0,
                    None, None)
        return ("DISTANCIA SEGURA", (0, 255, 0),
                "Sin interaccion", 0, 0,
                None, None)

# Inicia camara
camara = cv2.VideoCapture(0)
print("Legion IA v2 iniciado. Presiona Q para salir.")

while True:
    ret, frame = camara.read()
    if not ret:
        break

    frame_proceso = cv2.resize(frame, (480, 360))
    escala_x = frame.shape[1] / 480
    escala_y = frame.shape[0] / 360

    resultados = modelo(frame_proceso, conf=0.5, verbose=False)

    personas = []
    keypoints_lista = []

    for resultado in resultados:
        if resultado.keypoints is None:
            continue

        for i, caja in enumerate(resultado.boxes):
            x1, y1, x2, y2 = map(int, caja.xyxy[0])

            ancho = x2 - x1
            alto = y2 - y1
            if ancho < 80 or alto < 120:
                continue

            # Escala al frame original
            x1 = int(x1 * escala_x)
            y1 = int(y1 * escala_y)
            x2 = int(x2 * escala_x)
            y2 = int(y2 * escala_y)

            centro_x = (x1 + x2) // 2
            centro_y = (y1 + y2) // 2
            personas.append((centro_x, centro_y))

            kpts = resultado.keypoints.xy[i].cpu().numpy()
            keypoints_lista.append(kpts)

            # Caja del cuerpo
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                         (0, 255, 0), 2)

            # Dibuja nariz (cabeza) en azul mas grande
            if len(kpts) > 0:
                nx = int(kpts[0][0] * escala_x)
                ny = int(kpts[0][1] * escala_y)
                if nx > 0 and ny > 0:
                    cv2.circle(frame, (nx, ny), 5,
                              (255, 100, 0), -1)
                    cv2.putText(frame, "cabeza",
                                (nx+12, ny+5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 100, 0), 1)

            # Puntos de brazos en amarillo
            for punto_idx in [5, 6, 7, 8, 9, 10]:
                if punto_idx < len(kpts):
                    px = int(kpts[punto_idx][0] * escala_x)
                    py = int(kpts[punto_idx][1] * escala_y)
                    if px > 0 and py > 0:
                        cv2.circle(frame, (px, py), 5,
                                  (0, 255, 255), -1)

    # Analiza dos personas
    if len(personas) >= 2:
        p1 = personas[0]
        p2 = personas[1]

        distancia = ((p2[0]-p1[0])**2 + 
                     (p2[1]-p1[1])**2) ** 0.5

        if len(keypoints_lista) >= 2:
            resultado_accion = clasificar_accion(
                keypoints_lista[0],
                keypoints_lista[1],
                distancia,
                escala_x, escala_y
            )
            (accion, color, descripcion, 
             prob_saludo, dist_cabezas,
             nariz_p1, nariz_p2) = resultado_accion
        else:
            accion = "PRECAUCION"
            color = (0, 255, 255)
            descripcion = "Muy cercanos"
            prob_saludo = 0
            dist_cabezas = 0
            nariz_p1 = None
            nariz_p2 = None

        # Linea entre cuerpos
        cv2.line(frame, p1, p2, color, 2)

        # Linea entre cabezas si las detecta
        if nariz_p1 is not None and nariz_p2 is not None:
            n1 = (int(nariz_p1[0]), int(nariz_p1[1]))
            n2 = (int(nariz_p2[0]), int(nariz_p2[1]))
            cv2.line(frame, n1, n2, (255, 100, 0), 2)
            
            # Distancia entre cabezas en el punto medio
            mid_x = (n1[0] + n2[0]) // 2
            mid_y = (n1[1] + n2[1]) // 2
            cv2.putText(frame, 
                        f"Cabezas: {int(dist_cabezas)}px",
                        (mid_x-60, mid_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 100, 0), 1)

        # Puntos centrales cuerpo
        cv2.circle(frame, p1, 6, (0, 0, 255), -1)
        cv2.circle(frame, p2, 6, (0, 0, 255), -1)

        # Panel superior negro
        cv2.rectangle(frame, (0, 0),
                     (frame.shape[1], 130), (0, 0, 0), -1)

        # Etiqueta principal
        cv2.putText(frame, accion,
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, color, 3)

        # Descripcion
        cv2.putText(frame, descripcion,
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        # Probabilidad de saludo
        if prob_saludo > 0:
            bar_color = (0, 255, 0) if prob_saludo < 50 \
                        else (0, 165, 255) if prob_saludo < 75 \
                        else (0, 0, 255)
            cv2.putText(frame, 
                        f"Prob. saludo: {prob_saludo}%",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, bar_color, 2)

        # Distancia cuerpos
        cv2.putText(frame, 
                    f"Dist. cuerpos: {int(distancia)}px",
                    (frame.shape[1]-300, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    # Contador
    cv2.putText(frame, f"Personas: {len(personas)}",
                (20, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    cv2.putText(frame, "Legion IA v2",
                (frame.shape[1]-180, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (150, 150, 150), 1)

    cv2.imshow("Legion IA - Detector de Acciones", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()