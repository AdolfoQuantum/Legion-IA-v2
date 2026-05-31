import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from ultralytics import YOLO
from pathlib import Path

# ============================================================
# LEGION IA v3 — Detección en vivo con LSTM (optimizado CPU)
# ============================================================

BASE    = Path(r"C:\Users\accas\legion-ia")
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASES  = ["SEGURA", "PRECAUCION", "PELIGRO"]
COLORES = {
    "SEGURA":     (0, 200, 80),
    "PRECAUCION": (0, 180, 255),
    "PELIGRO":    (0, 0, 255),
}

IDX_CADERA_IZQ = 11
IDX_CADERA_DER = 12
IDX_HOMBRO_IZQ = 5
IDX_HOMBRO_DER = 6
FRAMES_SEQ     = 30

# ── Modelo LSTM ──────────────────────────────────────────────
class LegionLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=128,
                 num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True, dropout=0.3)
        self.bn   = nn.BatchNorm1d(hidden_size)
        self.fc1  = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2  = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.relu(self.fc1(out))
        out = self.drop(out)
        return self.fc2(out)

# ── Cargar modelos ───────────────────────────────────────────
print("Cargando modelos...")
yolo = YOLO(str(BASE / "yolov8n-pose.pt"))
lstm = LegionLSTM().to(DEVICE)
lstm.load_state_dict(torch.load(
    BASE / "modelos" / "legion_lstm_v2_best.pth",
    map_location=DEVICE))
lstm.eval()
print(f"✔ Modelos cargados — dispositivo: {DEVICE}")

# ── Normalización ────────────────────────────────────────────
def normalizar_keypoints(kpts):
    cadera = (kpts[IDX_CADERA_IZQ] + kpts[IDX_CADERA_DER]) / 2
    hombro = (kpts[IDX_HOMBRO_IZQ] + kpts[IDX_HOMBRO_DER]) / 2
    escala = np.linalg.norm(hombro - cadera) + 1e-6
    return ((kpts - cadera) / escala).flatten().astype(np.float32)

# ── Buffer por persona ───────────────────────────────────────
class BufferPersona:
    def __init__(self):
        self.frames    = deque(maxlen=FRAMES_SEQ)
        self.clase     = "..."
        self.confianza = 0.0
        self.color     = (150, 150, 150)

    def agregar(self, kpts_norm):
        self.frames.append(kpts_norm)

    def predecir(self, modelo):
        if len(self.frames) < 5:
            return
        seq = list(self.frames)
        while len(seq) < FRAMES_SEQ:
            seq.append(seq[-1])
        tensor = torch.tensor(
            np.array(seq), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out   = modelo(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            idx   = np.argmax(probs)
        self.clase     = CLASES[idx]
        self.confianza = probs[idx]
        self.color     = COLORES[self.clase]

buffers = {}

# ── Dibujado ─────────────────────────────────────────────────
def dibujar_persona(frame, x1, y1, x2, y2, kpts,
                    escala_x, escala_y, buf, pid):
    color = buf.color
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for idx in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        if idx < len(kpts):
            px = int(kpts[idx][0] * escala_x)
            py = int(kpts[idx][1] * escala_y)
            if px > 0 and py > 0:
                cv2.circle(frame, (px, py), 4, color, -1)

    if len(kpts) > 0:
        nx = int(kpts[0][0] * escala_x)
        ny = int(kpts[0][1] * escala_y)
        if nx > 0 and ny > 0:
            cv2.circle(frame, (nx, ny), 6, (255, 100, 0), -1)

    etiqueta = f"P{pid}: {buf.clase} {buf.confianza*100:.0f}%"
    ty = max(y1 - 10, 20)
    cv2.rectangle(frame, (x1, ty-20),
                  (x1 + len(etiqueta)*11, ty+5), (0,0,0), -1)
    cv2.putText(frame, etiqueta, (x1+4, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def dibujar_panel(frame, personas, buffers_activos):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.putText(frame, "LEGION IA v3",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (200, 200, 200), 2)
    cv2.putText(frame, f"Personas: {personas}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (150, 150, 150), 1)

    nivel = "SEGURA"
    for buf in buffers_activos:
        if buf.clase == "PELIGRO":
            nivel = "PELIGRO"
            break
        elif buf.clase == "PRECAUCION" and nivel != "PELIGRO":
            nivel = "PRECAUCION"

    cv2.putText(frame, f"ESTADO: {nivel}",
                (w-320, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, COLORES[nivel], 2)
    cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Q: salir  |  R: resetear buffers",
                (20, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (100, 100, 100), 1)

# ── Loop principal ───────────────────────────────────────────
camara = cv2.VideoCapture(0)
camara.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Legion IA v3 iniciado. Q=salir, R=resetear")
frame_count = 0

while True:
    ret, frame = camara.read()
    if not ret:
        break

    frame_count += 1

    # Resolución reducida para más velocidad
    frame_proc = cv2.resize(frame, (320, 240))
    escala_x   = frame.shape[1] / 416
    escala_y   = frame.shape[0] / 320

    resultados = yolo(frame_proc, conf=0.5, verbose=False)

    ids_frame       = set()
    buffers_activos = []

    for resultado in resultados:
        if resultado.keypoints is None:
            continue

        for i, caja in enumerate(resultado.boxes):
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            if (x2-x1) < 50 or (y2-y1) < 80:
                continue

            cx  = int((x1+x2)/2 * escala_x)
            cy  = int((y1+y2)/2 * escala_y)
            pid = f"{cx//80}_{cy//80}"
            ids_frame.add(pid)

            if pid not in buffers:
                buffers[pid] = BufferPersona()

            buf  = buffers[pid]
            kpts = resultado.keypoints.xy[i].cpu().numpy()

            if len(kpts) == 17:
                kpts_norm = normalizar_keypoints(kpts)
                buf.agregar(kpts_norm)

                # Predecir cada 8 frames para más fluidez
                if frame_count % 15 == 0:
                    buf.predecir(lstm)

            buffers_activos.append(buf)

            x1s = int(x1 * escala_x)
            y1s = int(y1 * escala_y)
            x2s = int(x2 * escala_x)
            y2s = int(y2 * escala_y)

            dibujar_persona(frame, x1s, y1s, x2s, y2s,
                           kpts, escala_x, escala_y, buf,
                           list(ids_frame).index(pid)+1)

    for pid in list(buffers.keys()):
        if pid not in ids_frame:
            del buffers[pid]

    dibujar_panel(frame, len(ids_frame), buffers_activos)

    cv2.imshow("Legion IA v3 — Deteccion en vivo", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('r'):
        buffers.clear()
        print("Buffers reseteados")

camara.release()
cv2.destroyAllWindows()
print("Legion IA v3 cerrado.")