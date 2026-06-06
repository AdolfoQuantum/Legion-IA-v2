import numpy as np
from pathlib import Path

# ============================================================
# LEGION IA — Dataset con 4 clases (VERDE/AZUL/AMARILLO/ROJO)
# ============================================================

BASE    = Path(r"C:\Users\accas\legion-ia")
DATASET = BASE / "modelos" / "dataset"

# ── Nueva taxonomía de 4 clases ─────────────────────────────
# Clase 0 = VERDE    → acciones completamente seguras
# Clase 1 = AZUL     → contacto amistoso o situación inusual
# Clase 2 = AMARILLO → empujones, caídas, forcejeo leve
# Clase 3 = ROJO     → peleas activas, golpes

CATEGORIAS = {
    # VERDE — acciones completamente seguras
    "saludos_verde": {
        "carpeta":  "saludos",
        "etiqueta": 0,
        "clases_ntu": ["A058", "A059", "A060", "A112", "A119"],
    },
    # AZUL — contacto amistoso o situación inusual
    "saludos_azul": {
        "carpeta":  "saludos",
        "etiqueta": 1,
        "clases_ntu": ["A055", "A053", "A002"],
    },
    # AMARILLO — empujones, caídas, forcejeo leve
    "neutros_amarillo": {
        "carpeta":  "neutros",
        "etiqueta": 2,
        "clases_ntu": ["A052", "A043", "A011", "A108"],
    },
    # ROJO — peleas activas, golpes
    "agresiones_rojo": {
        "carpeta":  "agresiones",
        "etiqueta": 3,
        "clases_ntu": ["A050", "A051"],
    },
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
        p1 = normalizar_persona(frame[0])
        p2 = normalizar_persona(frame[1])
        combinado = np.concatenate([p1, p2])  # 68 features
        secuencia.append(combinado)

    return np.array(secuencia, dtype=np.float32)  # (30, 68)

def extraer_codigo_accion(nombre_archivo):
    """Extrae el código de acción del nombre del archivo NTU"""
    # Formato: S001C001P001R001A050.skeleton
    nombre = nombre_archivo.stem.upper()
    try:
        idx_a = nombre.index('A')
        return nombre[idx_a:idx_a+4]  # Ej: A050
    except:
        return None

def procesar_categoria(nombre, config):
    carpeta  = BASE / "videos" / config["carpeta"] / "esqueletos"
    archivos = list(carpeta.glob("*.skeleton"))
    clases_validas = config["clases_ntu"]

    # Filtrar solo los archivos de las clases NTU correspondientes
    archivos_filtrados = [
        f for f in archivos
        if extraer_codigo_accion(f) in clases_validas
    ]

    if not archivos_filtrados:
        print(f"  ⚠ Sin archivos para {nombre} — clases: {clases_validas}")
        return [], []

    print(f"\n{'='*55}")
    print(f"  {nombre.upper()} — {len(archivos_filtrados)} archivos")
    print(f"  Clases: {clases_validas}")
    print(f"{'='*55}")

    X, y    = [], []
    errores = 0

    for i, archivo in enumerate(archivos_filtrados, 1):
        try:
            frames    = leer_skeleton(archivo)
            secuencia = normalizar_secuencia(frames)
            X.append(secuencia)
            y.append(config["etiqueta"])

            if i % 300 == 0 or i == len(archivos_filtrados):
                print(f"  [{i:04d}/{len(archivos_filtrados)}] procesados...")
        except Exception as e:
            errores += 1

    print(f"  ✔ {len(X)} secuencias OK — {errores} errores")
    return X, y

if __name__ == "__main__":
    print("🦴 LEGION IA — Dataset 4 clases (VERDE/AZUL/AMARILLO/ROJO)")
    print("="*55)

    X_total, y_total = [], []

    for nombre, config in CATEGORIAS.items():
        X, y = procesar_categoria(nombre, config)
        X_total.extend(X)
        y_total.extend(y)

    X_np = np.array(X_total)
    y_np = np.array(y_total)

    print(f"\n{'='*55}")
    print(f"  Shape X: {X_np.shape}")
    print(f"  Clase 0 VERDE:    {np.sum(y_np==0)} muestras")
    print(f"  Clase 1 AZUL:     {np.sum(y_np==1)} muestras")
    print(f"  Clase 2 AMARILLO: {np.sum(y_np==2)} muestras")
    print(f"  Clase 3 ROJO:     {np.sum(y_np==2)} muestras")
    print(f"  TOTAL:            {len(y_np)} muestras")
    print(f"{'='*55}")

    DATASET.mkdir(parents=True, exist_ok=True)
    np.save(DATASET / "X_skeleton_v4.npy", X_np)
    np.save(DATASET / "y_skeleton_v4.npy", y_np)

    print(f"\n✅ Guardado como X_skeleton_v4.npy e y_skeleton_v4.npy")
    print(f"   Tamaño: {X_np.nbytes/1024/1024:.1f} MB")