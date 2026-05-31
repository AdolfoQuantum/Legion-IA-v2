import cv2
from pathlib import Path

BASE = Path(r"C:\Users\accas\legion-ia\videos")
FORMATOS = {".mp4", ".webm", ".avi", ".mov", ".mpeg"}

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

if __name__ == "__main__":
    origen  = BASE / "peleas"
    destino = BASE / "peleas" / "frames"
    destino.mkdir(parents=True, exist_ok=True)

    videos = [f for f in origen.rglob("*")
              if f.is_file()
              and f.suffix.lower() in FORMATOS
              and "frames" not in str(f)]

    print(f"🎬 Videos encontrados: {len(videos)}")

    total = 0
    for i, video in enumerate(videos, 1):
        frames = extraer_frames(video, destino)
        total += frames
        if i % 100 == 0 or i == len(videos):
            print(f"  [{i:04d}/{len(videos)}] — {total:,} frames")

    print(f"\n✅ COMPLETO — {total:,} frames totales")