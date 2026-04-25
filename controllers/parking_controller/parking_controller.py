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
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
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

# ==================== GEMELO DIGITAL ====================
digital_twin = {}  # global_id -> info

def load_spot_attributes(config_path="parking_config.json"):
    """Carga type, priority desde el JSON y construye el digital_twin."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    global_id = config.get("spot_id_start", 1)
    for cam_data in config["cameras"]:
        cam_id = cam_data["id"]
        width = cam_data["spot_width"]
        height = cam_data["spot_height"]
        for spot in cam_data["spots"]:
            spot_type = spot.get("type", "normal")
            priority = spot.get("priority", 999)
            digital_twin[global_id] = {
                "camera": cam_id,
                "type": spot_type,
                "priority": priority,
                "status": "free",
                "x": spot["x"],
                "y": spot["y"],
                "width": width,
                "height": height
            }
            global_id += 1

load_spot_attributes()

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

# ==================== CONFIGURACIÓN ROUND‑ROBIN (original funcional) ====================
ANALYSIS_INTERVAL = 1.0
last_analysis_time = {}
num_cams = len(cameras)
start_time = robot.getTime()
for idx, cam_name in enumerate(cameras.keys()):
    offset = (idx / num_cams) * ANALYSIS_INTERVAL
    last_analysis_time[cam_name] = start_time - offset

accumulated_results = {}
last_send_time = start_time

print(f"Round‑robin activado: {num_cams} cámaras, cada una se analiza cada {ANALYSIS_INTERVAL} s")
print("Iniciando bucle principal...\n")

# ==================== FUNCIONES AUXILIARES ====================
def update_digital_twin_from_results(results):
    """Actualiza el estado del gemelo digital con los resultados de clasificación.
       No sobrescribe el estado 'reserved'."""
    for r in results:
        if digital_twin[r.global_id]["status"] != "reserved":
            digital_twin[r.global_id]["status"] = r.status

def assign_spot(zone_letter, required_type):
    """Elige la plaza más cercana (menor priority) de la zona indicada,
       que esté libre y coincida con el tipo requerido.
       La plaza seleccionada se marca como 'reserved'.
       Retorna el global_id o None si no hay disponible."""
    zone_to_cameras = {
        'A': ['cam_parking_A'],
        'B': ['cam_parking_B'],
        'C': ['cam_parking_CL'],
        'D': ['cam_parking_CR']
    }
    cam_names = zone_to_cameras.get(zone_letter)
    if not cam_names:
        return None

    candidates = []
    for gid, info in digital_twin.items():
        if info["camera"] in cam_names and info["status"] == "free":
            if required_type is None or info["type"] == required_type:
                candidates.append((gid, info["priority"]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    selected_id = candidates[0][0]
    digital_twin[selected_id]["status"] = "reserved"
    return selected_id

# ==================== BUCLE PRINCIPAL ====================
while robot.step(timestep) != -1:
    current_time = robot.getTime()

    # --- Procesar cámaras cuyo intervalo ha vencido (round‑robin) ---
    for cam_name, cam in cameras.items():
        if current_time - last_analysis_time[cam_name] >= ANALYSIS_INTERVAL:
            last_analysis_time[cam_name] = current_time

            img_data = cam.getImage()
            if not img_data:
                continue
            w = cam.getWidth()
            h = cam.getHeight()
            frame_bgra = np.frombuffer(img_data, np.uint8).reshape((h, w, 4))
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            spots = CAMERA_SPOTS[cam_name]
            if not spots:
                continue

            try:
                results = backend.classify_spots(frame_bgr, spots)
            except Exception as e:
                print(f"  Error en {cam_name}: {e}")
                continue

            # Actualizar gemelo digital (respetando reservas)
            update_digital_twin_from_results(results)

            # Guardar estado de plazas usando el estado actualizado del twin
            accumulated_results[cam_name] = [(r.global_id, digital_twin[r.global_id]["status"]) for r in results]

            if yolo:
                try:
                    detections = backend.detect_vehicles(frame_bgr)
                    detections_serializable = [
                        {
                            "bbox": list(d["bbox"]),
                            "confidence": d["confidence"],
                            "class_name": d["class_name"]
                        }
                        for d in detections
                    ]
                    accumulated_results[f"_detections_{cam_name}"] = detections_serializable
                except Exception as e:
                    print(f"  Error obteniendo detecciones YOLO en {cam_name}: {e}")

            img_b64 = encode_frame_to_base64(frame_bgra)
            accumulated_results[f"_image_{cam_name}"] = img_b64

            free_count = sum(1 for r in results if digital_twin[r.global_id]["status"] == "free")
            occ_count = len(results) - free_count
            print(f"[t={current_time:.2f}s] {cam_name}: {free_count} libres, {occ_count} ocupadas")

    # --- Envío periódico de resultados (cada 1 segundo) ---
    if current_time - last_send_time >= ANALYSIS_INTERVAL:
        if accumulated_results:
            # Añadir el gemelo digital completo al mensaje
            accumulated_results["digital_twin"] = digital_twin
            message = json.dumps(accumulated_results)
            robot.wwiSendText(message)
            accumulated_results = {}
        last_send_time = current_time

    # --- Atender mensajes del frontend (reservas) en cada paso ---
    # CORRECCIÓN: manejar None y procesar TODOS los mensajes pendientes
    while True:
        msg = robot.wwiReceiveText()
        if msg is None or msg == "":
            break
        try:
            req = json.loads(msg)
            if req.get("action") == "assign_spot":
                zone = req.get("zone")
                req_type = req.get("required_type")
                selected_id = assign_spot(zone, req_type)
                if selected_id is not None:
                    response = {
                        "action": "assign_spot_result",
                        "spot_id": selected_id,
                        "digital_twin": digital_twin
                    }
                else:
                    response = {
                        "action": "assign_spot_result",
                        "error": "No available spots",
                        "digital_twin": digital_twin
                    }
                robot.wwiSendText(json.dumps(response))
        except Exception as e:
            print(f"Error procesando mensaje: {e}", flush=True)

print("Controlador finalizado.")