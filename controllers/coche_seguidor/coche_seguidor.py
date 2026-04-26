from vehicle import Driver
import math
import time

driver = Driver()
TIME_STEP = int(driver.getBasicTimeStep())

dev = True

# =========================
# PARAMETROS
# =========================

DISTANCIA_FRENADO = 1.5
DISTANCIA_PARADA = 0.5

VELOCIDAD_CRUCERO = 6.0  # m/s

KP_STEER = 1.2
MAX_STEERING_ANGLE = math.radians(30)

TIMEOUT_SENAL = 2.0
DEBUG_LOG_INTERVAL = 0.5

# >>> REVERSA: manejar giros muy cerrados <<<
REVERSE_DURATION = 2.0              # segundos en marcha atrás
REVERSE_SPEED = -2.0                # m/s negativa = hacia atrás
REVERSE_ERROR_THRESHOLD = math.radians(150)   # 150° de error para activar
REVERSE_DISTANCE_THRESHOLD = 10.0   # solo si está a menos de 10 m del target

# =========================
# DISPOSITIVOS
# =========================

gps = driver.getDevice("gps")
gps.enable(TIME_STEP)

compass = driver.getDevice("compass")
compass.enable(TIME_STEP)

receiver = driver.getDevice("receiver")
receiver.enable(TIME_STEP)
receiver.setChannel(1)

print("coche_seguidor activo. Esperando objetivo del supervisor.")

# =========================
# ESTADO
# =========================

target_x = None
target_y = None

ultimo_mensaje = time.time()
last_debug_log = 0.0

previous_target_x = None
previous_target_y = None

# >>> REVERSA: temporizador <<<
reverse_until = 0.0

# =========================
# UTILS
# =========================

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def heading_from_compass(compass_values):
    return math.atan2(compass_values[1], compass_values[0])


# =========================
# LOOP PRINCIPAL
# =========================

while driver.step() != -1:

    # =========================
    # A. RECEPCION OBJETIVO
    # =========================
    while receiver.getQueueLength() > 0:
        mensaje = receiver.getString()
        receiver.nextPacket()
        try:
            partes = mensaje.split(',')
            new_target_x = float(partes[0])
            new_target_y = float(partes[1])
            if new_target_x != previous_target_x or new_target_y != previous_target_y:
                target_x = new_target_x
                target_y = new_target_y
                previous_target_x = target_x
                previous_target_y = target_y
                if dev:
                    print(f"[coche] NUEVO OBJETIVO RECIBIDO: ({target_x:.2f}, {target_y:.2f})")
            ultimo_mensaje = time.time()
        except Exception:
            print(f"Mensaje invalido: {mensaje}")

    # =========================
    # B. SEGURIDAD
    # =========================
    if target_x is None or target_y is None:
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(1.0)
        continue

    if (time.time() - ultimo_mensaje) > TIMEOUT_SENAL:
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(1.0)
        target_x = target_y = None
        continue

    # =========================
    # C. ESTADO ACTUAL
    # =========================
    pos = gps.getValues()
    mi_x, mi_y = pos[0], pos[1]

    dx = target_x - mi_x
    dy = target_y - mi_y
    distancia = math.hypot(dx, dy)

    brujula = compass.getValues()
    mi_angulo = normalize_angle(heading_from_compass(brujula))
    angulo_objetivo = math.atan2(dx, dy)
    error = normalize_angle(angulo_objetivo - mi_angulo)

    # >>> REVERSA: activar si el error es muy grande y estamos cerca <<<
    if (time.time() > reverse_until and
        abs(error) > REVERSE_ERROR_THRESHOLD and
        distancia < REVERSE_DISTANCE_THRESHOLD):
        reverse_until = time.time() + REVERSE_DURATION
        if dev:
            print("[coche] INICIANDO MANIOBRA DE MARCHA ATRÁS")

    # Si estamos dentro del tiempo de reversa, la aplicamos
    if time.time() < reverse_until:
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(REVERSE_SPEED)  # negativa → marcha atrás
        # El volante gira al máximo hacia el lado contrario del error
        steer_reverse = -math.copysign(MAX_STEERING_ANGLE, error)
        driver.setSteeringAngle(steer_reverse)
        if dev and (time.time() - last_debug_log) >= DEBUG_LOG_INTERVAL:
            last_debug_log = time.time()
            print(
                f"[coche] REVERSA pos=({mi_x:.2f},{mi_y:.2f}) "
                f"error={error:.2f} steer={steer_reverse:.2f} speed={REVERSE_SPEED:.2f}"
            )
        continue   # saltamos el control normal

    # =========================
    # D. STEERING (normal)
    # =========================
    steer_cmd = KP_STEER * math.tanh(error)
    angulo_volante = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, steer_cmd))
    driver.setSteeringAngle(angulo_volante)

    # =========================
    # E. VELOCIDAD
    # =========================

    # Velocidad base según alineación (0 = perpendicular, 1 = perfecto)
    alignment = max(0.0, 1.0 - abs(error) / math.pi)
    speed_cmd = VELOCIDAD_CRUCERO * alignment

    # Si el error es muy grande (>150°) y estamos cerca (<5 m), avance muy lento para no pasarnos
    if abs(error) > math.radians(150) and distancia < 5.0:
        speed_cmd = 1.0 / 3.6   # avance mínimo para poder girar
    # Si no, asegurar una velocidad mínima razonable para giros amplios (0.5 m/s)
    else:
        speed_cmd = max(speed_cmd, 1.5)   # al menos 1.5 m/s para poder girar bien

    # Evitar bloqueo total (ya cubierto por el mínimo anterior)
    # speed_cmd = max(speed_cmd, 0.5 / 3.6)  # sobra, lo quitamos

    # =========================
    # F. FRENADO (normal)
    # =========================
    if distancia < DISTANCIA_PARADA:
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(1.0)
        speed_cmd = 0.0
        brake_cmd = 1.0
    elif distancia < DISTANCIA_FRENADO:
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(0.4)
        speed_cmd = 0.0
        brake_cmd = 0.4
    else:
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(speed_cmd)
        brake_cmd = 0.0

    # =========================
    # G. DEBUG
    # =========================
    if dev and (time.time() - last_debug_log) >= DEBUG_LOG_INTERVAL:
        last_debug_log = time.time()
        print(
            f"[coche] pos=({mi_x:.2f}, {mi_y:.2f}) "
            f"target=({target_x:.2f}, {target_y:.2f}) "
            f"dist={distancia:.2f} "
            f"heading={mi_angulo:.2f} "
            f"target_ang={angulo_objetivo:.2f} "
            f"error={error:.2f} "
            f"steer={angulo_volante:.2f} "
            f"speed={speed_cmd:.2f} "
            f"brake={brake_cmd:.2f}"
        )