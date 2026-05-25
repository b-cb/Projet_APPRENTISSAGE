from ultralytics import YOLO
import cv2
import time
import torch
import threading

# ─── Configuration ────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.53
IMG_SIZE       = 320   # Réduire à 320 (vs 640 par défaut) = beaucoup plus rapide
WINDOW_NAME    = "Détection YOLOv11"

# ─── Détection du device ──────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
use_half = device == "cuda"   # FP16 uniquement sur GPU
print(f"Device : {device.upper()} | FP16 : {use_half} | imgsz : {IMG_SIZE}")

# ─── Chargement du modèle ─────────────────────────────────────────────────────
model = YOLO("./best.pt")
model.to(device)

# ─── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam.")
    exit()

# Forcer une résolution raisonnable pour la capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─── Thread de capture (évite que la webcam bloque l'inférence) ───────────────
latest_frame = [None]
frame_lock = threading.Lock()

def capture_loop():
    while cap.isOpened():
        ok, f = cap.read()
        if ok:
            with frame_lock:
                latest_frame[0] = f

capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()

# ─── Boucle principale ────────────────────────────────────────────────────────
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
conf_threshold = CONF_THRESHOLD
prev_time = time.time()

print("=== Détection YOLOv11 en temps réel ===")
print("  'q'       : quitter")
print("  '+' / '-' : augmenter / diminuer le seuil de confiance")

while True:
    # Récupérer le dernier frame disponible
    with frame_lock:
        frame = latest_frame[0]

    if frame is None:
        time.sleep(0.005)
        continue

    # Inférence
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        imgsz=IMG_SIZE,
        half=use_half,
        verbose=False,
    )

    r = results[0]
    annotated_frame = r.plot()

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    num_detections = len(r.boxes) if r.boxes is not None else 0

    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Objets: {num_detections}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Conf: {conf_threshold:.2f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(annotated_frame, device.upper(), (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    cv2.imshow(WINDOW_NAME, annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in (ord("+"), ord("=")):
        conf_threshold = round(min(0.99, conf_threshold + 0.05), 2)
        print(f"Conf : {conf_threshold:.2f}")
    elif key == ord("-"):
        conf_threshold = round(max(0.05, conf_threshold - 0.05), 2)
        print(f"Conf : {conf_threshold:.2f}")

cap.release()
cv2.destroyAllWindows()
