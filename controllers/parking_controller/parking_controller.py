from controller import Robot
import numpy as np
import json
import os
import sys
import base64
import cv2
import torch

from parking_detector import MorphologicalBackend, SpotConfig, YOLOBackend

yolo = True  # True usa Yolo, False usa morfológico

def encode_frame_to_base64(frame_bgra: np.ndarray) -> str:
    """Convierte un frame BGRA (Webots) a JPEG base64 para enviar al navegador."""
    # Convertir BGRA -> BGR
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    # Codificar como JPEG
    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

# ==================== CARGA DE CONFIGURACIÓN ====================
def load_parking_config(config_path="parking_config.json"):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(script_dir, config_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: No se encontró {full_path}")
        sys.exit(1)

    camera_spots = {}
    global_id = config.get("spot_id_start", 1)
    for cam_data in config["cameras"]:
        cam_id = cam_data["id"]
        width = cam_data["spot_width"]
        height = cam_data["spot_height"]
        spots = []
        for spot in cam_data["spots"]:
            spots.append(SpotConfig(
                global_id=global_id,
                x=spot["x"],
                y=spot["y"],
                width=width,
                height=height
            ))
            global_id += 1
        camera_spots[cam_id] = spots
    return camera_spots

print("Cargando configuración...")
CAMERA_SPOTS = load_parking_config("parking_config.json")
for cam, spots in CAMERA_SPOTS.items():
    print(f"  {cam}: {len(spots)} plazas")

# ==================== INICIALIZACIÓN ====================
robot = Robot()
timestep = int(robot.getBasicTimeStep())
print(f"Timestep básico: {timestep} ms")

# Habilitar cámaras
cameras = {}
for cam_name in CAMERA_SPOTS.keys():
    cam = robot.getDevice(cam_name)
    if cam is None:
        print(f"ERROR: No se encontró la cámara {cam_name}")
        continue
    cam.enable(timestep)
    cameras[cam_name] = cam
    print(f"Cámara {cam_name} habilitada. Resolución: {cam.getWidth()}x{cam.getHeight()}")

if not cameras:
    print("ERROR: No se pudo habilitar ninguna cámara. Saliendo.")
    sys.exit(1)

if not yolo:
    backend = MorphologicalBackend(free_threshold=900)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    backend = YOLOBackend(
        model_path="./model/yolov8n-visdrone.pt",
        mode="visdrone",
        ioa_threshold=0.4,
        iou_threshold=0.3,
        confidence_threshold=0.4,
        device=device
    )

# ==================== BUCLE PRINCIPAL (análisis a 1 Hz) ====================
analysis_interval_ms = 1000
steps_per_analysis = max(1, analysis_interval_ms // timestep)
print(f"Se analizará cada {steps_per_analysis} pasos de simulación (aprox. cada 1 segundo)")

step_counter = 0
last_time = robot.getTime()

while robot.step(timestep) != -1:
    step_counter += 1
    current_time = robot.getTime()

    if (current_time - last_time) < 1.0:
        continue

    last_time = current_time
    print(f"\n--- Análisis en t={current_time:.2f}s ---")

    all_results = {}

    for cam_name, cam in cameras.items():
        img_data = cam.getImage()
        if not img_data:
            continue
        w = cam.getWidth()
        h = cam.getHeight()
        frame_bgra = np.frombuffer(img_data, np.uint8).reshape((h, w, 4))
        # Convertir a BGR (3 canales) para los backends
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        spots = CAMERA_SPOTS[cam_name]
        if not spots:
            continue

        try:
            results = backend.classify_spots(frame_bgr, spots)
        except Exception as e:
            print(f"  Error en {cam_name}: {e}")
            continue

        # Guardar resultados de plazas
        all_results[cam_name] = [(r.global_id, r.status) for r in results]

        # --- Enviar detecciones YOLO (solo si está activo) ---
        if yolo:
            try:
                # Obtener detecciones de vehículos (modo visdrone)
                detections = backend.detect_vehicles(frame_bgr)
                # Convertir a formato serializable
                detections_serializable = [
                    {
                        "bbox": list(d["bbox"]),        # [x1, y1, x2, y2]
                        "confidence": d["confidence"],
                        "class_name": d["class_name"]
                    }
                    for d in detections
                ]
                all_results[f"_detections_{cam_name}"] = detections_serializable
            except Exception as e:
                print(f"  Error obteniendo detecciones YOLO en {cam_name}: {e}")

        # Enviar imagen de esta cámara
        img_b64 = encode_frame_to_base64(frame_bgra)
        all_results[f"_image_{cam_name}"] = img_b64

        free_count = sum(1 for r in results if r.status == "free")
        occ_count = len(results) - free_count
        print(f"  {cam_name}: {free_count} libres, {occ_count} ocupadas")

    if all_results:
        message = json.dumps(all_results)
        robot.wwiSendText(message)

print("Controlador finalizado.")