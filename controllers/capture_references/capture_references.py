"""
Temporary controller used to capture reference images
from the parking top-view cameras.

These reference frames are later used for parking
occupancy comparison and computer vision processing.
"""

from controller import Robot
import numpy as np
import cv2
import os

# Output directory where the captured reference
# images will be stored.
OUTPUT_DIR = "reference_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Webots robot controller
robot = Robot()

# Obtain simulation timestep
timestep = int(robot.getBasicTimeStep())

# Names of the parking cameras deployed
# across different parking zones.
CAMERA_NAMES = [
    "cam_parking_A",
    "cam_parking_B",
    "cam_parking_CL",
    "cam_parking_CR"
]

# Dictionary used to store initialized cameras
cameras = {}

# Initialize and enable all parking cameras
for name in CAMERA_NAMES:
    cam = robot.getDevice(name)

    # Skip camera if it is not available
    # in the current simulation world.
    if cam is None:
        print(f"⚠️ Camera '{name}' not found.")
        continue

    cam.enable(timestep)
    cameras[name] = cam

# Advance simulation one step to allow cameras
# to capture their first valid frame.
robot.step(timestep)

# Capture and store a reference image
# from each active camera.
for name, cam in cameras.items():

    # Obtain raw image buffer from Webots camera
    img_data = cam.getImage()

    if img_data is None:
        print(f"❌ Failed to capture image from {name}")
        continue

    # Retrieve camera resolution
    w = cam.getWidth()
    h = cam.getHeight()

    # Convert raw BGRA image buffer into
    # a NumPy array for OpenCV processing.
    frame = np.frombuffer(img_data, np.uint8).reshape((h, w, 4))

    # Convert image format from BGRA to BGR
    # since OpenCV operates using BGR encoding.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Generate output filename
    filename = os.path.join(OUTPUT_DIR, f"{name}_reference.png")

    # Save processed reference image
    success = cv2.imwrite(filename, frame_bgr)

    if success:
        print(f"✅ Reference image saved: {filename}")
    else:
        print(f"❌ Error saving {filename}")

print("Reference image capture completed.")