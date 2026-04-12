"""Controlador temporal para capturar una imagen de referencia de cada cámara cenital."""

from controller import Robot
import numpy as np
import cv2
import os

# Crear carpeta de salida si no existe
OUTPUT_DIR = "reference_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Lista de nombres de cámaras a capturar
CAMERA_NAMES = ["cam_parking_A", "cam_parking_B", "cam_parking_CL", "cam_parking_CR"]

# Habilitar todas las cámaras
cameras = {}
for name in CAMERA_NAMES:
    cam = robot.getDevice(name)
    if cam is None:
        print(f"⚠️ Cámara '{name}' no encontrada.")
        continue
    cam.enable(timestep)
    cameras[name] = cam

# Esperar un paso para que las cámaras capturen el primer frame
robot.step(timestep)

# Guardar cada imagen
for name, cam in cameras.items():
    img_data = cam.getImage()
    if img_data is None:
        print(f"❌ No se pudo obtener imagen de {name}")
        continue

    w = cam.getWidth()
    h = cam.getHeight()

    # Convertir de buffer (BGRA) a array numpy
    frame = np.frombuffer(img_data, np.uint8).reshape((h, w, 4))
    # Convertir BGRA → BGR para OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    filename = os.path.join(OUTPUT_DIR, f"{name}_reference.png")
    success = cv2.imwrite(filename, frame_bgr)
    if success:
        print(f"✅ Imagen guardada: {filename}")
    else:
        print(f"❌ Error al guardar {filename}")

print("Captura finalizada.")