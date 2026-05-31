import numpy as np
from pathlib import Path

# ============================================================
# LEGION IA — Dataset con 17 joints x 2 coords x 2 personas
# = 68 features (interaccion entre dos personas)
# ============================================================

BASE    = Path(r"C:\Users\accas\legion-ia")
DATASET = BASE / "modelos" / "dataset"

CATEGORIAS = {
    "agresiones": {"etiqueta": 2, "nivel": "PELIGRO"},
    "neutros":    {"etiqueta": 1, "nivel": "PRECAUCION"},
    "saludos":    {"etiqueta": 0, "nivel": "SEGURA"},
}

# Mapeo NTU (25 joints) → YOLO (17 joints)
NTU_A_YOLO = [
    3, 3, 3, 3, 3,
    4, 8, 5, 9, 6, 10,
    12, 16, 13, 17, 14, 18,
]

NUM_JOINTS = 17
FRAMES_SEQ = 30

def leer_skeleton(ruta):
    with open(ruta, 'r') as f:
        lineas = f.read().strip().split('\n')

    idx = 0
    num_frames = int(lineas[idx]); idx += 1
    frames = []

    for _ in range(num_frames):
        num_personas = int(lineas[idx]); idx += 1
        personas_frame = []

        for p in range(num_personas):
            idx += 1
            num_joints = int(lineas[idx]); idx += 1
            joints_25 = []

            for _ in range(num_joints):
                vals = lineas[idx].split(); idx += 1
                x, y = float(vals[0]), float(vals[1])
                joints_25.append([x, y])

            if len(joints_25) >= 19:
                joints_17 = np.array(
                    [joints_25[ntu_idx] for ntu_idx in NTU_A_YOLO],
                    dtype=np.float32)
                personas_frame.append(joints_17)

        # Siempre 2 personas — rellenar con ceros si falta
        while len(personas_frame) < 2:
            personas_frame.append(np.zeros((NUM_JOINTS, 2), dtype=np.float32))

        frames.append(personas_frame[:2])

    return frames

def normalizar_persona(joints):
    cadera = (joints[11] + joints[12]) / 2
    hombro = (joints[5]  + joints[6])  / 2
    escala = np.linalg.norm(hombro - cadera) + 1e-6
    return ((joints - cadera) / escala).flatten()

def normalizar_secuencia(frames):
    if len(frames) >= FRAMES_SEQ:
        indices = np.linspace(0, len(frames)-1, FRAMES_SEQ, dtype=int)
        frames  = [frames[i] for i in indices]
    else:
        while len(frames) < FRAMES_SEQ:
            frames.append(frames[-1])

    secuencia = []
    for frame in frames:
        p1 = normalizar_persona(frame[0])  # 34 features
        p2 = normalizar_persona(frame[1])  # 34 features
        combinado = np.concatenate([p1, p2])  # 68 features
        secuencia.append(combinado)

    return np.array(secuencia, dtype=np.float32)  # (30, 68)

def procesar_carpeta(nombre, info):
    carpeta  = BASE / "videos" / nombre / "esqueletos"
    archivos = list(carpeta.glob("*.skeleton"))

    if not archivos:
        print(f"  ⚠ Sin archivos en {nombre}\\esqueletos")
        return [], []

    print(f"\n{'='*50}")
    print(f"  {nombre.upper()} [{info['nivel']}] — {len(archivos)} archivos")
    print(f"{'='*50}")

    X, y   = [], []
    errores = 0

    for i, archivo in enumerate(archivos, 1):
        try:
            frames    = leer_skeleton(archivo)
            secuencia = normalizar_secuencia(frames)
            X.append(secuencia)
            y.append(info["etiqueta"])

            if i % 500 == 0 or i == len(archivos):
                print(f"  [{i:04d}/{len(archivos)}] procesados...")
        except Exception as e:
            errores += 1

    print(f"  ✔ {len(X)} secuencias OK — {errores} errores")
    return X, y

if __name__ == "__main__":
    print("🦴 LEGION IA — Dataset 68 features (2 personas x 17 joints x 2 coords)")
    print("="*50)

    X_total, y_total = [], []

    for nombre, info in CATEGORIAS.items():
        X, y = procesar_carpeta(nombre, info)
        X_total.extend(X)
        y_total.extend(y)

    X_np = np.array(X_total)
    y_np = np.array(y_total)

    print(f"\n{'='*50}")
    print(f"  Shape X: {X_np.shape}  ← (muestras, frames, 68 features)")
    print(f"  Shape y: {y_np.shape}")
    print(f"  Clase 0 SEGURA:     {np.sum(y_np==0)}")
    print(f"  Clase 1 PRECAUCION: {np.sum(y_np==1)}")
    print(f"  Clase 2 PELIGRO:    {np.sum(y_np==2)}")
    print(f"{'='*50}")

    DATASET.mkdir(parents=True, exist_ok=True)
    np.save(DATASET / "X_skeleton_v3.npy", X_np)
    np.save(DATASET / "y_skeleton_v3.npy", y_np)

    print(f"\n✅ Guardado como X_skeleton_v3.npy e y_skeleton_v3.npy")
    print(f"   Tamaño: {X_np.nbytes/1024/1024:.1f} MB")