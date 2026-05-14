from controller import Supervisor
import numpy as np
import json
import math
import os
import sys
import base64
import heapq
import cv2
import torch

from parking_detector import MorphologicalBackend, SpotConfig, YOLOBackend

yolo = True
dev = False

CRUISE_ALT = 3.0
HOVER_ALT = 3.0
HOME_ALT = 0.5
MIN_DIST = 1.5
ARRIVAL_RADIUS = 3.0
SPOT_ARRIVAL_RADIUS = 1.0
HOME_ARRIVAL_RADIUS = 1.0
CAR_PARK_RADIUS = 2.5            # margen tolerante: el coche frena 0.5-1.2 m antes del centro

# === Seguimiento fluido del coche ===
# El coche persigue un punto proyectado sobre la ruta a CAR_FOLLOW_DISTANCE metros
# por DETRAS del dron (medido a lo largo del path). De esa forma:
#  - si el dron va rapido, el target avanza rapido y el coche acelera
#  - si el dron se para (hover, ascenso), el target se queda fijo y el coche frena
#  - el coche nunca rebasa al dron porque su target esta SIEMPRE detras
CAR_WP_SPACING = 1.5             # densificacion de la ruta del coche (m entre puntos)
CAR_FOLLOW_DISTANCE = 8.0        # m de separacion ideal coche-dron a lo largo del path
CAR_FOLLOW_DISTANCE_FINAL = 0.0  # cuando mission_state pasa a "parking" se anula la separacion

# === Deteccion de coche aparcado ===
CAR_PARKED_DIST_TOL = 3.5        # m al centro de la plaza para considerar "candidato"
CAR_PARKED_MOVE_TOL = 0.05       # m de variacion maxima para considerarlo quieto
CAR_PARKED_HOLD_TIME = 1.5       # s parado consecutivos para confirmar aparcamiento


def encode_frame_to_base64(frame_bgra: np.ndarray) -> str:
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')


# ==================== PROYECCION PIXEL -> MUNDO ====================
def axis_angle_to_matrix(axis, angle):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def make_pixel_to_world(cam_cfg):
    cam_pos = np.array(cam_cfg["world_position"], dtype=float)
    R = axis_angle_to_matrix(cam_cfg["rotation"][:3], cam_cfg["rotation"][3])
    W = cam_cfg["image_width"]
    H = cam_cfg["image_height"]
    fov = cam_cfg["fov"]
    fx = W / (2.0 * math.tan(fov / 2.0))

    def project(px, py):
        dy = (px - W / 2.0) / fx
        dz = (py - H / 2.0) / fx
        ray = R @ np.array([-1.0, dy, dz])
        if abs(ray[2]) < 1e-9:
            return None
        t = -cam_pos[2] / ray[2]
        p = cam_pos + t * ray
        return float(p[0]), float(p[1])
    return project


# ==================== CARGA DE CONFIGURACION ====================
def load_parking_config(config_path="parking_config.json"):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(script_dir, config_path)
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)


print("Cargando configuracion...")
CONFIG = load_parking_config("parking_config.json")
CAMERA_SPOTS = {}
projectors = {}
digital_twin = {}

global_id = CONFIG.get("spot_id_start", 1)
for cam_data in CONFIG["cameras"]:
    cam_id = cam_data["id"]
    width = cam_data["spot_width"]
    height = cam_data["spot_height"]
    spots = []
    project = make_pixel_to_world(cam_data)
    projectors[cam_id] = project
    for spot in cam_data["spots"]:
        spots.append(SpotConfig(
            global_id=global_id,
            x=spot["x"],
            y=spot["y"],
            width=width,
            height=height
        ))
        cx_px = spot["x"] + width / 2.0
        cy_px = spot["y"] + height / 2.0
        wx, wy = project(cx_px, cy_px)
        digital_twin[global_id] = {
            "camera": cam_id,
            "type": spot.get("type", "normal"),
            "priority": spot.get("priority", 999),
            "status": "free",
            "x": spot["x"],
            "y": spot["y"],
            "width": width,
            "height": height,
            "world_x": wx,
            "world_y": wy,
        }
        global_id += 1
    CAMERA_SPOTS[cam_id] = spots
    print(f"  {cam_id}: {len(spots)} plazas")


# ==================== GRAFO DE CARRETERAS ====================
ROAD_NODES = {
    "home":          (-5.9,  -71.76),
    "access_south":  (0.0,  -71.77),
    "access_mid":    (0.0,  -52.65),
    "access_north":  (0.05,   0.12),
    "main_west_a":   (-24.0, 0.0),
    "main_east_b":   (68.0,  0.0),
    "A_entry":       (-24.0, 24.0),
    "B_entry":       (68.0,  24.0),
    "CL_entry":      (-15.0, -52.6),
    "CR_entry":      (15.0,  -52.3),
}
ROAD_EDGES = [
    ("home", "access_south"),
    ("access_south", "access_mid"),
    ("access_mid", "access_north"),
    ("access_north", "main_west_a"),
    ("access_north", "main_east_b"),
    ("main_west_a", "A_entry"),
    ("main_east_b", "B_entry"),
    ("access_mid", "CL_entry"),
    ("access_mid", "CR_entry"),
]
ZONE_TO_ENTRY = {
    "cam_parking_A":  "A_entry",
    "cam_parking_B":  "B_entry",
    "cam_parking_CL": "CL_entry",
    "cam_parking_CR": "CR_entry",
}


def build_adjacency():
    adj = {n: [] for n in ROAD_NODES}
    for a, b in ROAD_EDGES:
        adj[a].append(b)
        adj[b].append(a)
    return adj


ADJACENCY = build_adjacency()


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_node(xy):
    return min(ROAD_NODES, key=lambda n: euclid(ROAD_NODES[n], xy))


def astar(start_xy, goal_node):
    start_node = nearest_node(start_xy)
    if start_node == goal_node:
        return [ROAD_NODES[goal_node]], [goal_node]
    open_heap = [(0.0, start_node, [start_node])]
    visited = set()
    while open_heap:
        f, node, path = heapq.heappop(open_heap)
        if node == goal_node:
            return [ROAD_NODES[n] for n in path], path
        if node in visited:
            continue
        visited.add(node)
        for nb in ADJACENCY[node]:
            if nb in visited:
                continue
            g = sum(euclid(ROAD_NODES[path[i]], ROAD_NODES[path[i + 1]])
                    for i in range(len(path) - 1)) + euclid(ROAD_NODES[node], ROAD_NODES[nb])
            h = euclid(ROAD_NODES[nb], ROAD_NODES[goal_node])
            heapq.heappush(open_heap, (g + h, nb, path + [nb]))
    return [ROAD_NODES[goal_node]], [goal_node]


def build_flight_plan(start_xy, spot_world_xy, zone_entry_node):
    path_xy, path_names = astar(start_xy, zone_entry_node)
    print(f"  Ruta A*: {' -> '.join(path_names)}")
    waypoints = []
    waypoints.append([start_xy[0], start_xy[1], CRUISE_ALT])
    for nx, ny in path_xy:
        # Evitar duplicar el punto de inicio si ya está muy cerca
        if math.hypot(nx - start_xy[0], ny - start_xy[1]) > 0.5:
            waypoints.append([nx, ny, CRUISE_ALT])
    waypoints.append([spot_world_xy[0], spot_world_xy[1], CRUISE_ALT])
    waypoints.append([spot_world_xy[0], spot_world_xy[1], HOVER_ALT])
    return waypoints


def build_return_plan(current_xy, home_xy):
    path_xy, path_names = astar(current_xy, "home")
    print(f"  Ruta retorno A*: {' -> '.join(path_names)}")
    waypoints = [[current_xy[0], current_xy[1], CRUISE_ALT]]
    for nx, ny in path_xy:
        waypoints.append([nx, ny, CRUISE_ALT])
    waypoints.append([home_xy[0], home_xy[1], HOME_ALT])
    return waypoints


def densify_xy_path(points, max_spacing=CAR_WP_SPACING):
    """Devuelve una lista densificada de puntos (x, y) con separacion <= max_spacing.

    Mantiene los extremos y los nodos del grafo intactos, insertando puntos
    interpolados linealmente cuando la distancia entre dos consecutivos es mayor
    que max_spacing. Imprescindible para que el coche reciba targets continuos
    y no se quede esperando entre nodos del grafo de carreteras.
    """
    if not points:
        return []
    out = [tuple(points[0])]
    for i in range(1, len(points)):
        prev = out[-1]
        curr = tuple(points[i])
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        seg = math.hypot(dx, dy)
        if seg <= max_spacing or seg < 1e-6:
            out.append(curr)
            continue
        n_segments = int(math.ceil(seg / max_spacing))
        for j in range(1, n_segments):
            t = j / n_segments
            out.append((prev[0] + dx * t, prev[1] + dy * t))
        out.append(curr)
    return out


# ==================== SUPERVISOR ====================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
print(f"Timestep basico: {timestep} ms")

emitter_car = robot.getDevice("emitter_car")
emitter_car.setChannel(1)
emitter_drone = robot.getDevice("emitter_drone")
emitter_drone.setChannel(2)
drone_node = robot.getFromDef("DRONE")
car_node = robot.getFromDef("CAR")
if drone_node is None or car_node is None:
    print("ERROR: faltan DEF DRONE o DEF CAR en el world", flush=True)
    sys.exit(1)

drone_translation = drone_node.getField("translation")
car_translation = car_node.getField("translation")

HOME_XY = tuple(drone_translation.getSFVec3f()[:2])
print(f"Home dron: {HOME_XY}")

# ==================== VISUALIZACION GRAFO + RESERVA ====================
root_children = robot.getRoot().getField("children")


def _import_node(vrml):
    root_children.importMFNodeFromString(-1, vrml)


def visualize_road_graph():
    for name, (x, y) in ROAD_NODES.items():
        color = "0.1 0.7 1" if name != "home" else "1 0.7 0"
        _import_node(
            f'Solid {{ translation {x} {y} 0.3 name "_node_{name}" '
            f'children [ Shape {{ '
            f'appearance Appearance {{ material Material {{ diffuseColor {color} '
            f'transparency 0.2 }} }} '
            f'geometry Sphere {{ radius 0.6 }} }} ] }}'
        )
    for i, (a, b) in enumerate(ROAD_EDGES):
        ax, ay = ROAD_NODES[a]
        bx, by = ROAD_NODES[b]
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx) - math.pi / 2.0
        _import_node(
            f'Solid {{ translation {mx} {my} 0.15 rotation 0 0 1 {yaw} '
            f'name "_edge_{i}" '
            f'children [ Shape {{ '
            f'appearance Appearance {{ material Material {{ diffuseColor 0.1 0.7 1 '
            f'transparency 0.5 }} }} '
            f'geometry Cylinder {{ radius 0.15 height {length} }} }} ] }}'
        )

if dev:
    visualize_road_graph()
    print("Grafo de carreteras pintado.")

RESERVED_MARKER_NAME = "_reserved_marker"


def remove_reserved_marker():
    n = robot.getFromDef("RES_MARKER")
    if n is not None:
        n.remove()


def show_reserved_marker(xy):
    remove_reserved_marker()
    x, y = xy
    _import_node(
        f'DEF RES_MARKER Solid {{ translation {x} {y} 0.05 '
        f'name "{RESERVED_MARKER_NAME}" children [ Shape {{ '
        f'appearance Appearance {{ material Material {{ diffuseColor 1 0.1 0.1 '
        f'emissiveColor 1 0 0 }} }} '
        f'geometry Cylinder {{ radius 1.2 height 0.05 }} }} ] }}'
    )


def drone_xy():
    p = drone_translation.getSFVec3f()
    return (p[0], p[1])


def drone_xyz():
    p = drone_translation.getSFVec3f()
    return (p[0], p[1], p[2])


def car_xy():
    p = car_translation.getSFVec3f()
    return (p[0], p[1])


def send_drone_waypoints(wps):
    msg = json.dumps({"waypoints": wps})
    emitter_drone.send(msg.encode('utf-8'))
    print(f"Enviados {len(wps)} waypoints al dron")


def send_car_target(xy):
    msg = f"{xy[0]:.3f},{xy[1]:.3f}"
    emitter_car.send(msg.encode('utf-8'))


# ==================== CAMARAS ====================
cameras = {}
for cam_name in CAMERA_SPOTS.keys():
    cam = robot.getDevice(cam_name)
    if cam is None:
        print(f"ERROR: No se encontro la camara {cam_name}")
        continue
    cam.enable(timestep)
    cameras[cam_name] = cam

if not cameras:
    print("ERROR: ninguna camara habilitada"); sys.exit(1)

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

ANALYSIS_INTERVAL = 1.0
last_analysis_time = {}
num_cams = len(cameras)
start_time = robot.getTime()
for idx, cam_name in enumerate(cameras.keys()):
    offset = (idx / num_cams) * ANALYSIS_INTERVAL
    last_analysis_time[cam_name] = start_time - offset

accumulated_results = {}
last_send_time = start_time


# ==================== MISION ====================
mission_state = "idle"
current_spot_id = None
spot_world_xy = None
car_waypoints = []          # (x,y) DENSIFICADOS para el coche (desde access_south en adelante)
drone_waypoints = []        # [x,y,z] completo (sin duplicados)
drone_wp_index = 0          # índice del dron en drone_waypoints
current_car_target = None   # destino actual enviado al coche
access_south_idx = None     # se calculará en start_mission

# Estado del seguimiento fluido del coche
drone_path_idx = 0          # proyeccion (no decreciente) de la xy del dron sobre car_waypoints

# Estado para deteccion de coche aparcado (estacionario cerca de la plaza)
car_last_xy = None
car_still_since = None

# Métricas de tiempo
mission_start_time = 0.0
drone_time = 0.0
car_time = 0.0


ZONE_TO_CAMERAS = {
    'A': ['cam_parking_A'],
    'B': ['cam_parking_B'],
    'C': ['cam_parking_CL'],
    'D': ['cam_parking_CR'],
}


def assign_spot(zone_letter, required_type):
    """Selecciona la mejor plaza libre de la zona pedida que cumpla el tipo.

    Devuelve (selected_id, error). Si selected_id es None, error es un string
    descriptivo ("invalid_zone", "no_type_in_zone", "all_occupied").
    Acepta required_type en {None, "normal", "electric", "handicap"}; las
    cadenas vacias se tratan como None (cualquier tipo).
    """
    if required_type in ("", "any"):
        required_type = None

    cam_names = ZONE_TO_CAMERAS.get(zone_letter)
    if not cam_names:
        print(f"[assign] zona '{zone_letter}' desconocida")
        return None, "invalid_zone"

    # Plazas de la zona (independientemente del estado) para distinguir errores
    zone_spots = [(gid, info) for gid, info in digital_twin.items()
                  if info["camera"] in cam_names]

    # Filtrar por tipo (si se pide uno concreto)
    if required_type is None:
        type_spots = zone_spots
    else:
        type_spots = [(gid, info) for gid, info in zone_spots
                      if info["type"] == required_type]

    if not type_spots:
        print(f"[assign] zona {zone_letter}: no hay plazas de tipo '{required_type}'")
        return None, "no_type_in_zone"

    candidates = [(gid, info["priority"]) for gid, info in type_spots
                  if info["status"] == "free"]
    if not candidates:
        print(f"[assign] zona {zone_letter} tipo {required_type}: todas ocupadas/reservadas")
        return None, "all_occupied"

    candidates.sort(key=lambda x: x[1])
    selected_id = candidates[0][0]
    digital_twin[selected_id]["status"] = "reserved"
    print(f"[assign] zona {zone_letter} tipo {required_type} -> plaza {selected_id} "
          f"(priority {digital_twin[selected_id]['priority']})")
    return selected_id, None


def update_digital_twin_from_results(results):
    for r in results:
        if digital_twin[r.global_id]["status"] == "reserved":
            # Solo pisar "reserved" si la camara ve un vehiculo Y nuestro coche esta lejos
            # (evita falso positivo cuando el propio coche entra a la plaza)
            if r.status == "occupied" and current_spot_id == r.global_id:
                cx, cy = car_xy()
                d = math.hypot(cx - digital_twin[r.global_id]["world_x"],
                               cy - digital_twin[r.global_id]["world_y"])
                if d > CAR_PARK_RADIUS * 2:
                    digital_twin[r.global_id]["status"] = "occupied"
                    print(f"[digital_twin] Plaza {r.global_id} reservada marcada occupied por camara (intruso detectado)")
            continue
        digital_twin[r.global_id]["status"] = r.status


def start_mission(spot_id):
    global mission_state, current_spot_id, spot_world_xy
    global drone_waypoints, car_waypoints, drone_wp_index, current_car_target, access_south_idx
    global drone_path_idx, car_last_xy, car_still_since
    global mission_start_time, drone_time, car_time

    mission_start_time = robot.getTime()
    drone_time = 0.0
    car_time = 0.0

    info = digital_twin[spot_id]
    spot_world_xy = (info["world_x"], info["world_y"])
    zone_entry = ZONE_TO_ENTRY[info["camera"]]
    print(f"Mision iniciada -> plaza {spot_id} en {spot_world_xy}")

    plan = build_flight_plan(drone_xy(), spot_world_xy, zone_entry)
    drone_waypoints = plan

    # Encontrar access_south en el plan (tolerancia 0.5 m)
    access_south_xy = ROAD_NODES["access_south"]
    access_south_idx = None
    for i, wp in enumerate(drone_waypoints):
        if math.hypot(wp[0] - access_south_xy[0], wp[1] - access_south_xy[1]) < 0.5:
            access_south_idx = i
            break
    if access_south_idx is None:
        print("ERROR: no se encontró access_south en el plan de vuelo")
        access_south_idx = 1   # fallback

    # Ruta del coche desde access_south en adelante, DENSIFICADA para seguimiento fluido
    raw_car_path = [(pt[0], pt[1]) for pt in drone_waypoints[access_south_idx:]]
    car_waypoints = densify_xy_path(raw_car_path, max_spacing=CAR_WP_SPACING)
    print(f"  Ruta coche: {len(raw_car_path)} nodos -> {len(car_waypoints)} waypoints densificados")

    drone_wp_index = 0
    drone_path_idx = 0
    car_last_xy = None
    car_still_since = None
    # Inicialmente el target esta al inicio de la ruta del coche (zona de salida)
    current_car_target = car_waypoints[0]

    send_drone_waypoints(drone_waypoints)
    if dev:
        show_reserved_marker(spot_world_xy)

    current_spot_id = spot_id
    mission_state = "approaching"


def redirect_to_spot(new_spot_id):
    """Redirige la mision en curso hacia una nueva plaza SIN reiniciar la ruta.

    Solo sustituye los dos ultimos waypoints del dron (crucero sobre plaza +
    hover) y el tramo final del car_waypoints a partir del nodo de entrada de
    zona. Todo lo anterior (nodos de grafo ya recorridos o en camino) se conserva,
    de modo que el dron y el coche no se "teletransportan" ni pierden su contexto.

    Asume que la nueva plaza esta en la misma zona (mismo zone_entry_node),
    que es el caso habitual al reasignar por intruso.
    """
    global current_spot_id, spot_world_xy
    global drone_waypoints, car_waypoints
    global car_last_xy, car_still_since

    info = digital_twin[new_spot_id]
    new_spot_xy = (info["world_x"], info["world_y"])

    # =========================
    # POSICION ACTUAL DEL DRON
    # =========================
    dx, dy, dz = drone_xyz()

    # Buscar waypoint actual mas cercano
    closest_idx = min(
        range(len(drone_waypoints)),
        key=lambda i: math.hypot(
            drone_waypoints[i][0] - dx,
            drone_waypoints[i][1] - dy
        )
    )

    # Mantener SOLO la parte futura de la ruta
    remaining_route = drone_waypoints[closest_idx:]

    # Evitar pequeños retrocesos
    if len(remaining_route) > 0:
        d0 = math.hypot(
            remaining_route[0][0] - dx,
            remaining_route[0][1] - dy
        )
        if d0 < 2.0:
            remaining_route = remaining_route[1:]

    # Eliminar el destino viejo
    if len(remaining_route) >= 2:
        remaining_route = remaining_route[:-2]

    # Nueva ruta desde posicion REAL actual
    drone_waypoints = (
        [[dx, dy, dz]]
        + remaining_route
        + [
            [new_spot_xy[0], new_spot_xy[1], CRUISE_ALT],
            [new_spot_xy[0], new_spot_xy[1], HOVER_ALT],
        ]
    )

    # =========================
    # COCHE
    # =========================
    zone_entry_node = ZONE_TO_ENTRY[info["camera"]]
    entry_xy = ROAD_NODES[zone_entry_node]

    entry_idx = min(
        range(len(car_waypoints)),
        key=lambda i: math.hypot(
            car_waypoints[i][0] - entry_xy[0],
            car_waypoints[i][1] - entry_xy[1]
        )
    )

    new_tail = densify_xy_path(
        [car_waypoints[entry_idx], new_spot_xy],
        max_spacing=CAR_WP_SPACING
    )

    car_waypoints = list(car_waypoints[:entry_idx]) + new_tail

    # =========================
    # ESTADO
    # =========================
    spot_world_xy = new_spot_xy
    current_spot_id = new_spot_id
    car_last_xy = None
    car_still_since = None

    send_drone_waypoints(drone_waypoints)

    if dev:
        show_reserved_marker(spot_world_xy)

    print(
        f"[redirect] Redirigido a plaza {new_spot_id} "
        f"desde posicion actual del dron."
    )

def reassign_spot():
    """Busca otra plaza con los mismos criterios y redirige la mision hacia ella.
    Devuelve True si se encontro alternativa, False si no hay ninguna libre."""
    global current_spot_id

    info = digital_twin[current_spot_id]
    cam_name = info["camera"]
    required_type = info.get("type")

    zone_letter = None
    for letter, cams in ZONE_TO_CAMERAS.items():
        if cam_name in cams:
            zone_letter = letter
            break

    # Marcar la plaza ocupada por el intruso
    digital_twin[current_spot_id]["status"] = "occupied"
    current_spot_id = None

    if zone_letter is None:
        print("[reassign] No se pudo determinar la zona.")
        return False

    new_id, err = assign_spot(zone_letter, required_type)
    if new_id is None:
        print(f"[reassign] Sin alternativa en zona {zone_letter} tipo {required_type}: {err}")
        return False

    print(f"[reassign] Nueva plaza: {new_id}")
    redirect_to_spot(new_id)
    return True


def _project_xy_onto_path(xy, path, start_idx=0):
    """Indice del waypoint mas cercano a xy a partir de start_idx (no retrocede)."""
    best_i = start_idx
    best_d = float('inf')
    # Pequena ventana de busqueda hacia adelante para acelerar
    for i in range(start_idx, len(path)):
        d = math.hypot(xy[0] - path[i][0], xy[1] - path[i][1])
        if d < best_d:
            best_d = d
            best_i = i
        elif d > best_d + CAR_WP_SPACING * 4:
            break
    return best_i


def _point_back_along_path(path, ref_idx, distance_back):
    """Devuelve un (x, y) que esta `distance_back` metros antes de path[ref_idx]
    siguiendo el path hacia atras. Si no hay suficiente longitud, devuelve path[0]."""
    if ref_idx <= 0 or distance_back <= 0:
        return path[max(0, ref_idx)]
    remaining = distance_back
    i = ref_idx
    while i > 0 and remaining > 0:
        ax, ay = path[i - 1]
        bx, by = path[i]
        seg = math.hypot(bx - ax, by - ay)
        if seg <= 0:
            i -= 1
            continue
        if seg >= remaining:
            t = remaining / seg
            return (bx - (bx - ax) * t, by - (by - ay) * t)
        remaining -= seg
        i -= 1
    return path[0]


def track_drone_and_update_car():
    """Avanza el indice del dron por waypoints (solo a efectos de logging interno)."""
    global drone_wp_index
    if mission_state != "approaching":
        return
    if drone_wp_index >= len(drone_waypoints) - 1:
        return
    target_wp = drone_waypoints[drone_wp_index + 1]
    tx, ty, _ = target_wp
    dx, dy, _ = drone_xyz()
    dist = math.hypot(dx - tx, dy - ty)
    if dist < ARRIVAL_RADIUS:
        drone_wp_index += 1


def update_car_following():
    """Actualiza current_car_target cada step.

    Estrategia: el coche persigue un punto sobre el path densificado situado
    CAR_FOLLOW_DISTANCE metros DETRAS de la proyeccion del dron sobre el path.

    - Mientras el dron avanza, el target avanza con el (separacion ~constante).
    - Si el dron se queda quieto (ascenso, hover, frenazo), el target se queda
      fijo y el coche frena por su propia logica de DISTANCIA_FRENADO/PARADA.
    - El target NUNCA esta por delante del dron, asi que el coche no rebasa.
    - En "parking"/"returning" el target pasa a ser la plaza (ultimo wp).
    """
    global current_car_target, drone_path_idx

    if mission_state == "idle" or not car_waypoints:
        return

    if mission_state == "approaching":
        dx, dy, _ = drone_xyz()
        drone_path_idx = _project_xy_onto_path((dx, dy), car_waypoints, drone_path_idx)
        # Si el dron aun no ha entrado en el path del coche (esta despegando),
        # mandamos al coche al primer wp con calma.
        if drone_path_idx <= 0:
            current_car_target = car_waypoints[0]
            return
        current_car_target = _point_back_along_path(
            car_waypoints, drone_path_idx, CAR_FOLLOW_DISTANCE
        )
    else:
        # parking / returning: el coche entra a la plaza.
        current_car_target = car_waypoints[-1]


def update_mission():
    global mission_state, current_spot_id, current_car_target, access_south_idx
    global drone_wp_index, drone_path_idx
    global car_last_xy, car_still_since
    global drone_time, car_time

    if mission_state == "idle":
        return
    dx, dy, dz = drone_xyz()

    # ── Deteccion de intruso ──────────────────────────────────────────────────
    # Si la plaza reservada aparece como "occupied" y nuestro coche aun esta
    # lejos (no somos nosotros los que la estamos ocupando), es un intruso.
    if mission_state in ("approaching", "parking") and current_spot_id is not None:
        cx, cy = car_xy()
        d_car_to_spot = math.hypot(cx - spot_world_xy[0], cy - spot_world_xy[1])
        if d_car_to_spot > CAR_PARK_RADIUS * 2:
            if digital_twin[current_spot_id]["status"] == "occupied":
                print(f"[intruso] Plaza {current_spot_id} ocupada. Reasignando...")
                if not reassign_spot():
                    print("[intruso] Sin alternativas. Dron retorna a casa.")
                    mission_state = "returning"
                    send_drone_waypoints(build_return_plan((dx, dy), HOME_XY))
                return
    # ─────────────────────────────────────────────────────────────────────────

    if mission_state == "approaching":
        target = (spot_world_xy[0], spot_world_xy[1], HOVER_ALT)
        d_xy = math.hypot(dx - target[0], dy - target[1])
        d_z = abs(dz - target[2])
        if d_xy < SPOT_ARRIVAL_RADIUS and d_z < 1.0:
            drone_time = robot.getTime() - mission_start_time
            print(f"Dron sobre plaza en {drone_time:.2f}s. Iniciando retorno y esperando al coche.")
            mission_state = "parking"
            # Reiniciar tracking de "coche quieto"
            car_last_xy = None
            car_still_since = None
            send_drone_waypoints(build_return_plan((dx, dy), HOME_XY))
    elif mission_state == "parking":
        cx, cy = car_xy()
        d_car = math.hypot(cx - spot_world_xy[0], cy - spot_world_xy[1])
        now = robot.getTime()

        # Criterio 1: el coche entra dentro del radio "amplio" del spot
        parked_by_radius = d_car < CAR_PARK_RADIUS

        # Criterio 2: el coche se ha quedado quieto cerca de la plaza
        parked_by_still = False
        if d_car < CAR_PARKED_DIST_TOL:
            if car_last_xy is None:
                car_last_xy = (cx, cy)
                car_still_since = now
            else:
                moved = math.hypot(cx - car_last_xy[0], cy - car_last_xy[1])
                if moved < CAR_PARKED_MOVE_TOL:
                    if car_still_since is not None and (now - car_still_since) >= CAR_PARKED_HOLD_TIME:
                        parked_by_still = True
                else:
                    # se ha movido, reiniciar contador
                    car_last_xy = (cx, cy)
                    car_still_since = now
        else:
            car_last_xy = None
            car_still_since = None

        if parked_by_radius or parked_by_still:
            car_time = robot.getTime() - mission_start_time
            reason = "radio" if parked_by_radius else "estacionario"
            print(f"Coche aparcado en plaza {current_spot_id} (criterio: {reason}, d={d_car:.2f} m) en {car_time:.2f}s.")
            digital_twin[current_spot_id]["status"] = "occupied"
            mission_state = "returning"
    elif mission_state == "returning":
        if math.hypot(dx - HOME_XY[0], dy - HOME_XY[1]) < HOME_ARRIVAL_RADIUS and dz < HOME_ALT + 0.3:
            print("Dron en base. Mision completada.")
            remove_reserved_marker()
            mission_state = "idle"
            current_spot_id = None
            car_waypoints.clear()
            drone_waypoints.clear()
            drone_wp_index = 0
            current_car_target = None
            access_south_idx = None
            drone_path_idx = 0
            car_last_xy = None
            car_still_since = None


print("Iniciando bucle principal...\nMesaje importante: para mandar el dron y el coche a la plaza, seleccionar una zona en la interfaz web.\n")

while robot.step(timestep) != -1:
    current_time = robot.getTime()

    for cam_name, cam in cameras.items():
        if current_time - last_analysis_time[cam_name] < ANALYSIS_INTERVAL:
            continue
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

        update_digital_twin_from_results(results)
        accumulated_results[cam_name] = [
            (r.global_id, digital_twin[r.global_id]["status"]) for r in results
        ]

        if yolo:
            try:
                detections = backend.detect_vehicles(frame_bgr)
                accumulated_results[f"_detections_{cam_name}"] = [
                    {"bbox": list(d["bbox"]), "confidence": d["confidence"], "class_name": d["class_name"]}
                    for d in detections
                ]
            except Exception as e:
                print(f"  Error YOLO {cam_name}: {e}")

        accumulated_results[f"_image_{cam_name}"] = encode_frame_to_base64(frame_bgra)

    if current_time - last_send_time >= ANALYSIS_INTERVAL:
        if accumulated_results:
            accumulated_results["digital_twin"] = digital_twin
            accumulated_results["mission_state"] = mission_state
            accumulated_results["metrics"] = {
                "mission_start_time": mission_start_time,
                "drone_time": drone_time,
                "car_time": car_time,
                "current_time": current_time
            }
            robot.wwiSendText(json.dumps(accumulated_results))
            accumulated_results = {}
        last_send_time = current_time

    track_drone_and_update_car()
    update_car_following()
    update_mission()

    if current_car_target is not None:
        send_car_target(current_car_target)

    while True:
        msg = robot.wwiReceiveText()
        if msg is None or msg == "":
            break
        if not msg.startswith("{"):
            continue
        try:
            req = json.loads(msg)
        except Exception:
            continue
        try:
            if req.get("action") == "assign_spot":
                if mission_state != "idle":
                    response = {"action": "assign_spot_result",
                                "error": "drone_busy",
                                "digital_twin": digital_twin}
                    robot.wwiSendText(json.dumps(response))
                    continue
                zone = req.get("zone")
                req_type = req.get("required_type")
                print(f"[supervisor] Solicitud: zona={zone} tipo={req_type}")
                selected_id, err = assign_spot(zone, req_type)
                if selected_id is not None:
                    start_mission(selected_id)
                    response = {"action": "assign_spot_result",
                                "spot_id": selected_id,
                                "zone": zone,
                                "required_type": req_type,
                                "digital_twin": digital_twin}
                else:
                    # err in {"invalid_zone", "no_type_in_zone", "all_occupied"}
                    response = {"action": "assign_spot_result",
                                "error": err or "no_available_spots",
                                "zone": zone,
                                "required_type": req_type,
                                "digital_twin": digital_twin}
                robot.wwiSendText(json.dumps(response))
        except Exception as e:
            print(f"Error procesando mensaje: {e}", flush=True)

print("Controlador finalizado.")