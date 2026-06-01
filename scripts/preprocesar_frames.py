import cv2
from pathlib import Path

# ============================================================
# LEGION IA — Extractor de frames v3 (peleas + neutros)
# ============================================================

BASE = Path(r"C:\Users\accas\legion-ia\videos")
FORMATOS = {".mp4", ".webm", ".avi", ".mov", ".mpeg"}

CONFIG = {
    "peleas": {
        "origen":     BASE / "peleas",
        "destino":    BASE / "peleas" / "frames",
        "fps":        5,
        "max_frames": 150,
    },
    "neutros": {
        "origen":     BASE / "neutros",
        "destino":    BASE / "neutros" / "frames",
        "fps":        5,
        "max_frames": 150,
    },
}

def extraer_frames(video_path, destino, fps_objetivo=5, max_frames=150):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 25
    intervalo = max(1, int(fps_video / fps_objetivo))
    carpeta_video = destino / video_path.stem
    carpeta_video.mkdir(parents=True, exist_ok=True)
    frame_num = 0
    guardados = 0
    while guardados < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % intervalo == 0:
            cv2.imwrite(str(carpeta_video / f"frame_{guardados:04d}.jpg"),
                       frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            guardados += 1
        frame_num += 1
    cap.release()
    return guardados

def procesar_categoria(nombre, config):
    origen  = config["origen"]
    destino = config["destino"]
    destino.mkdir(parents=True, exist_ok=True)

    videos = [f for f in origen.rglob("*")
              if f.is_file()
              and f.suffix.lower() in FORMATOS
              and "frames" not in str(f)]

    if not videos:
        print(f"\n  ⚠ Sin videos en {nombre}")
        return 0

    print(f"\n{'='*50}")
    print(f"  {nombre.upper()} — {len(videos)} videos")
    print(f"{'='*50}")

    total = 0
    for i, video in enumerate(videos, 1):
        frames = extraer_frames(video, destino,
                               config["fps"], config["max_frames"])
        total += frames
        if i % 100 == 0 or i == len(videos):
            print(f"  [{i:04d}/{len(videos)}] — {total:,} frames")

    print(f"\n  ✔ Total: {total:,} frames de {nombre}")
    return total

if __name__ == "__main__":
    print("🎬 LEGION IA — Extractor de frames v3")
    print("="*50)

    total_global = 0
    for nombre, config in CONFIG.items():
        total_global += procesar_categoria(nombre, config)

    print(f"\n{'='*50}")
    print(f"✅ COMPLETO — {total_global:,} frames totales")
    print(f"{'='*50}")