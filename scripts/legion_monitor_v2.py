import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from threading import Thread, Lock
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap

# ============================================================
# LEGION IA Vision v3 — Monitor multicámara 4 clases
# ============================================================

RUTA_LOGO  = r"C:\Users\accas\legion-ia\logo_legion.PNG"
RUTA_YOLO  = r"C:\Users\accas\legion-ia\yolov8n-pose.pt"
RUTA_LSTM  = r"C:\Users\accas\legion-ia\modelos\legion_lstm_v4_best.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_SEQ = 30

UMBRALES = {
    "VERDE":    0.70,
    "AZUL":     0.75,
    "AMARILLO": 0.80,
    "ROJO":     0.85,
}

COLORES = {
    "VERDE":    {"border": "#639922", "bg": "#0d1f0d", "text": "#639922"},
    "AZUL":     {"border": "#378ADD", "bg": "#0d1420", "text": "#378ADD"},
    "AMARILLO": {"border": "#EF9F27", "bg": "#1a1400", "text": "#EF9F27"},
    "ROJO":     {"border": "#E24B4A", "bg": "#1a0000", "text": "#E24B4A"},
}

COLORES_CV = {
    "VERDE":    (57,  200, 80),
    "AZUL":     (221, 138, 55),
    "AMARILLO": (39,  180, 255),
    "ROJO":     (74,  75,  226),
}

PRIORIDAD = ["VERDE", "AZUL", "AMARILLO", "ROJO"]

IDX_CADERA_IZQ = 11
IDX_CADERA_DER = 12
IDX_HOMBRO_IZQ = 5
IDX_HOMBRO_DER = 6

# ── Modelo LSTM v4 ───────────────────────────────────────────
class LegionLSTM(nn.Module):
    def __init__(self, input_size=68, hidden_size=128,
                 num_layers=2, num_classes=4):
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
yolo_model  = YOLO(RUTA_YOLO)
lstm_model  = LegionLSTM().to(DEVICE)
lstm_model.load_state_dict(torch.load(RUTA_LSTM, map_location=DEVICE))
lstm_model.eval()
modelo_lock = Lock()
print(f"Modelos cargados -- {DEVICE}")

# ── Utilidades ───────────────────────────────────────────────
def normalizar_keypoints(kpts):
    cadera = (kpts[IDX_CADERA_IZQ] + kpts[IDX_CADERA_DER]) / 2
    hombro = (kpts[IDX_HOMBRO_IZQ] + kpts[IDX_HOMBRO_DER]) / 2
    escala = np.linalg.norm(hombro - cadera) + 1e-6
    return ((kpts - cadera) / escala).flatten().astype(np.float32)

def detectar_caida(boxes):
    for i, caja in enumerate(boxes):
        if i >= 2:
            break
        x1, y1, x2, y2 = map(int, caja.xyxy[0])
        ancho = x2 - x1
        alto  = y2 - y1
        ratio = alto / (ancho + 1e-6)
        if ratio < 0.5:
            return True
    return False

def aplicar_umbrales(probs):
    mapa  = {0: "VERDE", 1: "AZUL", 2: "AMARILLO", 3: "ROJO"}
    idx   = np.argmax(probs)
    conf  = probs[idx]
    nivel = mapa[idx]
    while nivel != "VERDE" and conf < UMBRALES[nivel]:
        idx_actual = PRIORIDAD.index(nivel)
        nivel = PRIORIDAD[idx_actual - 1]
    return nivel, conf

def crear_logo_coloreado(ruta, color, size=55):
    img_cv = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if img_cv is None:
        return None
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv
    invertida = 255 - gray
    _, mascara = cv2.threshold(invertida, 20, 255, cv2.THRESH_BINARY)
    h, w = gray.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    rgba[:, :, 0] = r
    rgba[:, :, 1] = g
    rgba[:, :, 2] = b
    rgba[:, :, 3] = mascara
    rgba = np.ascontiguousarray(rgba)
    qimg = QImage(rgba.data, w, h, 4*w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg).scaled(
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def detectar_camaras(max_check=10):
    camaras = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                camaras.append(i)
            cap.release()
    return camaras

def get_nivel_inicial(idx, total):
    if total == 1: return "VERDE"
    p = idx / total
    if p < 0.25:   return "VERDE"
    if p < 0.50:   return "AZUL"
    if p < 0.75:   return "AMARILLO"
    return "ROJO"

# ── Analizador por cámara ────────────────────────────────────
class AnalizadorCamara(QObject):
    nivel_actualizado = pyqtSignal(int, str, float, object)

    def __init__(self, cam_id, cam_idx):
        super().__init__()
        self.cam_id       = cam_id
        self.cam_idx      = cam_idx
        self.activo       = True
        self.buffer       = deque(maxlen=FRAMES_SEQ)
        self.frame_actual = None
        self.lock         = Lock()
        self.cap          = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        self.hilo         = Thread(target=self._analizar, daemon=True)
        self.hilo.start()

    def _analizar(self):
        frame_count = 0
        while self.activo:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_count += 1
            with self.lock:
                self.frame_actual = frame.copy()

            if frame_count % 10 != 0:
                continue

            try:
                frame_small = cv2.resize(frame, (416, 320))
                with modelo_lock:
                    resultados = yolo_model(frame_small,
                                           conf=0.5, verbose=False)

                ultimo_resultado = None
                for resultado in resultados:
                    ultimo_resultado = resultado
                    if resultado.keypoints is None:
                        continue

                    kpts_todas = []
                    for j in range(min(2, len(resultado.boxes))):
                        k = resultado.keypoints.xy[j].cpu().numpy()
                        if len(k) == 17:
                            kpts_todas.append(normalizar_keypoints(k))

                    while len(kpts_todas) < 2:
                        kpts_todas.append(np.zeros(34, dtype=np.float32))

                    self.buffer.append(np.concatenate(kpts_todas))

                if len(self.buffer) >= 10:
                    seq = list(self.buffer)
                    while len(seq) < FRAMES_SEQ:
                        seq.append(seq[-1])
                    tensor = torch.tensor(
                        np.array(seq), dtype=torch.float32
                    ).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        out   = lstm_model(tensor)
                        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

                    nivel, conf = aplicar_umbrales(probs)

                    if ultimo_resultado is not None:
                        if detectar_caida(ultimo_resultado.boxes):
                            if PRIORIDAD.index(nivel) < PRIORIDAD.index("AMARILLO"):
                                nivel = "AMARILLO"
                                conf  = 0.90

                    frame_anot = self._anotar_frame(
                        frame, ultimo_resultado, nivel, conf)
                    self.nivel_actualizado.emit(
                        self.cam_id, nivel, conf, frame_anot)

            except Exception:
                pass

    def _anotar_frame(self, frame, resultado_yolo, nivel, conf):
        color = COLORES_CV.get(nivel, (100, 100, 100))
        h, w  = frame.shape[:2]

        if resultado_yolo is not None:
            for i, caja in enumerate(resultado_yolo.boxes):
                if i >= 2:
                    break
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                sx = w / 416
                sy = h / 320
                x1 = int(x1*sx); y1 = int(y1*sy)
                x2 = int(x2*sx); y2 = int(y2*sy)

                ancho = x2 - x1
                alto  = y2 - y1
                caida = (alto / (ancho + 1e-6)) < 0.5

                color_p  = COLORES_CV["AMARILLO"] if caida else color
                etiqueta = f"P{i+1}: CAIDA" if caida else \
                           f"P{i+1}: {nivel} {conf*100:.0f}%"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color_p, 2)
                ty = max(y1 - 8, 20)
                cv2.rectangle(frame, (x1, ty-18),
                              (x1 + len(etiqueta)*9, ty+4),
                              (0, 0, 0), -1)
                cv2.putText(frame, etiqueta, (x1+3, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color_p, 1)

                if resultado_yolo.keypoints is not None and \
                   i < len(resultado_yolo.keypoints.xy):
                    kpts = resultado_yolo.keypoints.xy[i].cpu().numpy()
                    for idx in [5, 6, 7, 8, 9, 10]:
                        if idx < len(kpts):
                            px = int(kpts[idx][0] * sx)
                            py = int(kpts[idx][1] * sy)
                            if px > 0 and py > 0:
                                cv2.circle(frame, (px, py), 4, color_p, -1)

        # Panel superior sin caracteres especiales
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
        texto_nivel = f"LEGION IA  {nivel}  {conf*100:.0f}%"
        ahora = datetime.now().strftime("%Y/%m/%d  %H:%M:%S")
        cv2.putText(frame, texto_nivel,
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2)
        cv2.putText(frame, ahora,
                    (w - 210, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (180, 180, 180), 1)

        return frame

    def get_frame(self):
        with self.lock:
            return self.frame_actual.copy() \
                if self.frame_actual is not None else None

    def detener(self):
        self.activo = False
        if self.cap:
            self.cap.release()

# ── Widget cámara pequeña ────────────────────────────────────
class CamaraWidget(QWidget):
    clicked = pyqtSignal(int, str)

    def __init__(self, cam_id, nivel_inicial, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.nivel  = nivel_inicial
        self.color  = COLORES[nivel_inicial]
        self.setFixedSize(70, 55)
        self.setCursor(Qt.PointingHandCursor)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.color['bg']};
                border: 2px solid {self.color['border']};
                border-radius: 5px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("border:none; background:transparent;")
        self.lbl_video.setFixedSize(66, 36)

        self.lbl_nombre = QLabel(f"CAM {self.cam_id:02d}")
        self.lbl_nombre.setAlignment(Qt.AlignCenter)
        self.lbl_nombre.setStyleSheet(
            f"color:{self.color['text']}; font-size:8px; "
            f"font-weight:bold; border:none; background:transparent;")

        layout.addWidget(self.lbl_video)
        layout.addWidget(self.lbl_nombre)

    def actualizar_nivel(self, nivel, frame=None):
        if nivel == self.nivel and frame is None:
            return
        self.nivel = nivel
        self.color = COLORES[nivel]
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.color['bg']};
                border: 2px solid {self.color['border']};
                border-radius: 5px;
            }}
        """)
        self.lbl_nombre.setStyleSheet(
            f"color:{self.color['text']}; font-size:8px; "
            f"font-weight:bold; border:none; background:transparent;")
        if frame is not None:
            self._mostrar_frame(frame)

    def _mostrar_frame(self, frame):
        try:
            f = cv2.resize(frame, (66, 36))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w, ch = f.shape
            img = QImage(f.data, w, h, ch*w, QImage.Format_RGB888)
            self.lbl_video.setPixmap(QPixmap.fromImage(img))
        except:
            pass

    def mousePressEvent(self, event):
        self.clicked.emit(self.cam_id, self.color['border'])

# ── Widget cámara principal ──────────────────────────────────
class PrincipalWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.border_color  = "#639922"
        self.cam_id_actual = 1
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: #111111;
                border: 3px solid {self.border_color};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("border:none; background:transparent;")
        self.lbl_video.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.lbl_info = QLabel("CAM 01  VERDE")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setFixedHeight(22)
        self.lbl_info.setStyleSheet(
            f"color:{self.border_color}; font-size:12px; "
            f"font-weight:bold; border:none; background:transparent;")

        layout.addWidget(self.lbl_video)
        layout.addWidget(self.lbl_info)

    def actualizar(self, cam_id, nivel, conf, frame):
        color = COLORES[nivel]['border']
        self.border_color  = color
        self.cam_id_actual = cam_id
        self.setStyleSheet(f"""
            QWidget {{
                background-color: #111111;
                border: 3px solid {color};
                border-radius: 8px;
            }}
        """)
        self.lbl_info.setText(
            f"CAM {cam_id:02d}  {nivel}  {conf*100:.0f}%")
        self.lbl_info.setStyleSheet(
            f"color:{color}; font-size:12px; "
            f"font-weight:bold; border:none; background:transparent;")

        if frame is not None:
            try:
                w = self.lbl_video.width()
                h = self.lbl_video.height()
                if w > 0 and h > 0:
                    f = cv2.resize(frame, (w, h))
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    fh, fw, ch = f.shape
                    img = QImage(f.data, fw, fh, ch*fw,
                                 QImage.Format_RGB888)
                    self.lbl_video.setPixmap(QPixmap.fromImage(img))
            except:
                pass

    def seleccionar_camara(self, cam_id, color):
        self.cam_id_actual = cam_id
        self.border_color  = color
        self.setStyleSheet(f"""
            QWidget {{
                background-color: #111111;
                border: 3px solid {color};
                border-radius: 8px;
            }}
        """)
        self.lbl_info.setStyleSheet(
            f"color:{color}; font-size:12px; "
            f"font-weight:bold; border:none; background:transparent;")

# ── Logo con 4 colores ───────────────────────────────────────
class LogoWidget(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pulso        = True
        self.nivel_maximo = "VERDE"
        self.setFixedHeight(70)
        self.setCursor(Qt.PointingHandCursor)
        self._setup_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self._pulsar)
        self.timer.start(700)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(2)
        self.setStyleSheet("background:transparent;")

        self.logo_lbl = QLabel()
        self.logo_lbl.setAlignment(Qt.AlignCenter)
        self.logo_lbl.setStyleSheet("border:none; background:transparent;")

        pixmap = crear_logo_coloreado(RUTA_LOGO, "#639922")
        if pixmap:
            self.logo_lbl.setPixmap(pixmap)
            self.tiene_imagen = True
        else:
            self.logo_lbl.setText("LEGION")
            self.logo_lbl.setStyleSheet(
                "color:#639922; font-size:20px; font-weight:bold; "
                "border:none; background:transparent;")
            self.tiene_imagen = False

        self.sub_lbl = QLabel("Presiona para ver mayor alerta")
        self.sub_lbl.setAlignment(Qt.AlignCenter)
        self.sub_lbl.setStyleSheet(
            "color:#639922; font-size:9px; border:none; background:transparent;")

        layout.addWidget(self.logo_lbl)
        layout.addWidget(self.sub_lbl)

    def set_nivel_maximo(self, nivel):
        self.nivel_maximo = nivel

    def _pulsar(self):
        color_base = COLORES[self.nivel_maximo]['border']
        pulsa = self.nivel_maximo in ["AMARILLO", "ROJO"]

        if pulsa:
            self.pulso = not self.pulso
            color = color_base if self.pulso else "#0a0a0f"
        else:
            color = color_base

        if self.tiene_imagen:
            pixmap = crear_logo_coloreado(RUTA_LOGO, color)
            if pixmap:
                self.logo_lbl.setPixmap(pixmap)
        else:
            self.logo_lbl.setStyleSheet(
                f"color:{color}; font-size:20px; font-weight:bold; "
                "border:none; background:transparent;")
        self.sub_lbl.setStyleSheet(
            f"color:{color}; font-size:9px; "
            "border:none; background:transparent;")

    def mousePressEvent(self, event):
        self.clicked.emit()

# ── Ventana principal ────────────────────────────────────────
class LegionMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Legion  IA Vision")
        self.setStyleSheet("background-color: #0a0a0f;")
        self.resize(960, 640)

        self.cam_seleccionada = 1
        self.cam_widgets      = {}
        self.analizadores     = {}
        self.niveles_actuales = {}

        print("Detectando camaras...")
        self.indices_camaras = detectar_camaras()
        self.num_camaras     = len(self.indices_camaras)
        print(f"Camaras: {self.num_camaras} -> {self.indices_camaras}")

        self._setup_ui()
        self._iniciar_analizadores()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        header = QHBoxLayout()
        titulo = QLabel("Legion  IA Vision")
        titulo.setStyleSheet(
            "color:#e0e0e0; font-size:14px; font-weight:bold;")
        n = self.num_camaras
        self.lbl_estado = QLabel(
            f"  {n} camara{'s' if n>1 else ''} activa{'s' if n>1 else ''}")
        self.lbl_estado.setStyleSheet("color:#639922; font-size:11px;")
        header.addWidget(titulo)
        header.addStretch()
        header.addWidget(self.lbl_estado)
        main_layout.addLayout(header)

        if n == 0:
            lbl = QLabel("No se detectaron camaras")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#555; font-size:16px;")
            main_layout.addWidget(lbl)

        elif n == 1:
            self.principal = PrincipalWidget()
            main_layout.addWidget(self.principal)

        else:
            left_count  = max(1, round(n * 0.25))
            right_count = max(1, round(n * 0.25))
            top_count   = n - left_count - right_count

            niveles_ini   = [get_nivel_inicial(i, n) for i in range(n)]
            left_niveles  = niveles_ini[:left_count]
            top_niveles   = niveles_ini[left_count:left_count+top_count]
            right_niveles = niveles_ini[left_count+top_count:]

            top_row = QHBoxLayout()
            top_row.setSpacing(3)
            esp_izq = QWidget()
            esp_izq.setFixedWidth(70)
            top_row.addWidget(esp_izq)
            for i, nivel in enumerate(top_niveles):
                cam_id = left_count + i + 1
                w = CamaraWidget(cam_id, nivel)
                w.clicked.connect(self.on_cam_click)
                top_row.addWidget(w)
                self.cam_widgets[cam_id] = w
                self.niveles_actuales[cam_id] = nivel
            esp_der = QWidget()
            esp_der.setFixedWidth(70)
            top_row.addWidget(esp_der)
            main_layout.addLayout(top_row)

            mid_row = QHBoxLayout()
            mid_row.setSpacing(3)

            col_izq = QVBoxLayout()
            col_izq.setSpacing(3)
            for i, nivel in enumerate(reversed(left_niveles)):
                cam_id = left_count - i
                w = CamaraWidget(cam_id, nivel)
                w.clicked.connect(self.on_cam_click)
                col_izq.addWidget(w)
                self.cam_widgets[cam_id] = w
                self.niveles_actuales[cam_id] = nivel
            col_izq_widget = QWidget()
            col_izq_widget.setFixedWidth(70)
            col_izq_widget.setLayout(col_izq)
            mid_row.addWidget(col_izq_widget)

            self.principal = PrincipalWidget()
            mid_row.addWidget(self.principal)

            col_der = QVBoxLayout()
            col_der.setSpacing(3)
            for i, nivel in enumerate(right_niveles):
                cam_id = left_count + top_count + i + 1
                w = CamaraWidget(cam_id, nivel)
                w.clicked.connect(self.on_cam_click)
                col_der.addWidget(w)
                self.cam_widgets[cam_id] = w
                self.niveles_actuales[cam_id] = nivel
            col_der_widget = QWidget()
            col_der_widget.setFixedWidth(70)
            col_der_widget.setLayout(col_der)
            mid_row.addWidget(col_der_widget)

            main_layout.addLayout(mid_row)

        self.logo = LogoWidget()
        self.logo.clicked.connect(self.on_logo_click)
        main_layout.addWidget(self.logo, alignment=Qt.AlignHCenter)

        barra = QHBoxLayout()
        leyenda_widget = QWidget()
        leyenda_layout = QHBoxLayout(leyenda_widget)
        leyenda_layout.setContentsMargins(0, 0, 0, 0)
        leyenda_layout.setSpacing(12)
        for texto, color in [
            ("Normal",     "#639922"),
            ("Extrano",    "#378ADD"),
            ("Precaucion", "#EF9F27"),
            ("Peligro",    "#E24B4A"),
        ]:
            lbl = QLabel(f"  {texto}")
            lbl.setStyleSheet(f"color:{color}; font-size:10px;")
            leyenda_layout.addWidget(lbl)

        info = QLabel(
            f"Legion IA Vision  "
            f"{n} camara{'s' if n>1 else ''} activa{'s' if n>1 else ''}")
        info.setStyleSheet("color:#444; font-size:10px;")
        barra.addWidget(leyenda_widget)
        barra.addStretch()
        barra.addWidget(info)
        main_layout.addLayout(barra)

    def _iniciar_analizadores(self):
        for i, cam_idx in enumerate(self.indices_camaras):
            cam_id = i + 1
            analizador = AnalizadorCamara(cam_id, cam_idx)
            analizador.nivel_actualizado.connect(self.on_nivel_actualizado)
            self.analizadores[cam_id] = analizador

    def on_nivel_actualizado(self, cam_id, nivel, conf, frame):
        if cam_id in self.cam_widgets:
            self.cam_widgets[cam_id].actualizar_nivel(nivel, frame)
            self.niveles_actuales[cam_id] = nivel

        if cam_id == self.cam_seleccionada:
            self.principal.actualizar(cam_id, nivel, conf, frame)

        nivel_max = "VERDE"
        for v in self.niveles_actuales.values():
            if PRIORIDAD.index(v) > PRIORIDAD.index(nivel_max):
                nivel_max = v
        self.logo.set_nivel_maximo(nivel_max)

        mensajes = {
            "VERDE":    (f"  {self.num_camaras} camaras activas", "#639922"),
            "AZUL":     ("  Movimiento inusual detectado",         "#378ADD"),
            "AMARILLO": ("  PRECAUCION  Revisar camaras",          "#EF9F27"),
            "ROJO":     ("  ALERTA  PELIGRO DETECTADO",            "#E24B4A"),
        }
        texto, color = mensajes.get(
            nivel_max,
            (f"  {self.num_camaras} camaras activas", "#639922"))
        self.lbl_estado.setText(texto)
        self.lbl_estado.setStyleSheet(
            f"color:{color}; font-size:11px; font-weight:bold;")

    def on_cam_click(self, cam_id, color):
        self.cam_seleccionada = cam_id
        self.principal.seleccionar_camara(cam_id, color)

    def on_logo_click(self):
        nivel_max = "VERDE"
        cam_max   = 1
        for cam_id, nivel in self.niveles_actuales.items():
            if PRIORIDAD.index(nivel) > PRIORIDAD.index(nivel_max):
                nivel_max = nivel
                cam_max   = cam_id
        color = COLORES[nivel_max]['border']
        self.cam_seleccionada = cam_max
        self.principal.seleccionar_camara(cam_max, color)

    def closeEvent(self, event):
        for analizador in self.analizadores.values():
            analizador.detener()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = LegionMonitor()
    ventana.show()
    sys.exit(app.exec_())