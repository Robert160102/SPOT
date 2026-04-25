import math
import os
from controller import Robot

# --- CONFIGURACIÓN DE VUELO ---
ALTURA_VUELO = 2.5       
VELOCIDAD_MAX = 0.1      
TOLERANCIA_DIST = 0.5

# --- CONSTANTES PID ---
K_VERTICAL_THRUST = 68.5    
K_VERTICAL_OFFSET = 0.6     
K_VERTICAL_P = 5.0         # Ascenso suave pero firme
K_ROLL_P = 50.0
K_PITCH_P = 30.0

K_POS = 0.02
K_VEL = 0.1

def cargar_ruta(fichero):
    waypoints = []
    if not os.path.exists(fichero):
        return waypoints
    with open(fichero, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith('Rotation') or linea.startswith('#') or 'angle' in linea:
                continue
            try:
                partes = linea.split()
                if len(partes) >= 3:
                    waypoints.append([float(partes[0]), float(partes[1]), float(partes[2])])
            except ValueError:
                pass
    return waypoints

def calcular_punto_extra(wp_penultimo, wp_ultimo, distancia=1.0):
    dx = wp_ultimo[0] - wp_penultimo[0]
    dz = wp_ultimo[2] - wp_penultimo[2]
    magnitud = math.sqrt(dx**2 + dz**2)
    if magnitud == 0: return wp_ultimo
    return [wp_ultimo[0] + (dx / magnitud) * distancia, wp_ultimo[1], wp_ultimo[2] + (dz / magnitud) * distancia]

def clamp(valor, limite):
    return max(-limite, min(limite, valor))

def calcular_altitude_input(error_altura):
    clamped = max(-1.0, min(1.0, error_altura + K_VERTICAL_OFFSET))
    return K_VERTICAL_P * (clamped ** 3)

def main():
    global ALTURA_VUELO

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    # 1. Inicializar Motores
    motores = []
    nombres_motores = ["front left propeller", "front right propeller", "rear left propeller", "rear right propeller"]
    for nombre in nombres_motores:
        motor = robot.getDevice(nombre)
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)
        motores.append(motor)
    m_front_left, m_front_right, m_rear_left, m_rear_right = motores

    # 2. Inicializar Sensores
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    waypoints = cargar_ruta('ruta.md')
    if len(waypoints) >= 2:
        waypoints.append(calcular_punto_extra(waypoints[-2], waypoints[-1], distancia=1.0))

    def velocidad_segura(v):
        return max(0.0, min(150.0, v))

    # --- FASE 1: CALIBRACIÓN DE SENSORES ---
    print("Encendiendo sistemas y calibrando...")
    # Fundamental: Dejar que la física se asiente y los sensores se estabilicen
    for _ in range(50):
        robot.step(timestep)
        
    pos_actual = gps.getValues()
    if not math.isnan(pos_actual[0]):
        ALTURA_VUELO = pos_actual[2] + 2.5
        print(f"Despegue autorizado. Altura objetivo: {ALTURA_VUELO:.2f}m")

    # --- FASE 2: VUELO CONTINUO ---
    current_wp_idx = 0
    modo_hover = False
    last_pos = gps.getValues()

    while robot.step(timestep) != -1:
        pos = gps.getValues()
        if math.isnan(pos[0]): continue

        vx = (pos[0] - last_pos[0]) / dt
        vz = (pos[1] - last_pos[1]) / dt
        last_pos = pos

        roll, pitch, yaw = imu.getRollPitchYaw()
        gyro_vals = gyro.getValues()
        roll_vel, pitch_vel, _ = gyro_vals[0], gyro_vals[1], gyro_vals[2]

        # --- LÓGICA DE NAVEGACIÓN ---
        error_altura = ALTURA_VUELO - pos[2]
        
        target_roll = 0.0
        target_pitch = 0.0
        
        # Moverse hacia los waypoints solo si está estable en el aire (ya ha subido)
        if error_altura < 1.0 and current_wp_idx < len(waypoints):
            target = waypoints[current_wp_idx]
            dx = target[0] - pos[0]
            dz = target[2] - pos[1]
            dist_horizontal = math.sqrt(dx**2 + dz**2)

            if dist_horizontal < TOLERANCIA_DIST:
                if current_wp_idx < len(waypoints) - 1:
                    current_wp_idx += 1
                    print(f"Waypoint {current_wp_idx} alcanzado.")
                elif not modo_hover:
                    print("Destino final. Modo Hover.")
                    modo_hover = True

            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            err_x_local = dx * cos_yaw + dz * sin_yaw
            err_z_local = -dx * sin_yaw + dz * cos_yaw
            vx_local = vx * cos_yaw + vz * sin_yaw
            vz_local = -vx * sin_yaw + vz * cos_yaw

            target_roll = clamp(-err_x_local * K_POS + vx_local * K_VEL, VELOCIDAD_MAX)
            target_pitch = clamp(-err_z_local * K_POS + vz_local * K_VEL, VELOCIDAD_MAX)

        # --- PID DE ESTABILIZACIÓN ---
        altitude_input = calcular_altitude_input(error_altura)

        roll_input = clamp(K_ROLL_P * (target_roll - roll) - roll_vel, 15.0)
        pitch_input = clamp(K_PITCH_P * (target_pitch - pitch) - pitch_vel, 15.0)

        motor_fl_vel = K_VERTICAL_THRUST + altitude_input - roll_input - pitch_input
        motor_fr_vel = K_VERTICAL_THRUST + altitude_input + roll_input - pitch_input
        motor_rl_vel = K_VERTICAL_THRUST + altitude_input - roll_input + pitch_input
        motor_rr_vel = K_VERTICAL_THRUST + altitude_input + roll_input + pitch_input

        # ¡SIGNOS ORIGINALES RESTAURADOS! (-, +, +, -) para que todos generen sustentación positiva
        m_front_left.setVelocity(-velocidad_segura(motor_fl_vel))
        m_front_right.setVelocity(velocidad_segura(motor_fr_vel))
        m_rear_left.setVelocity(velocidad_segura(motor_rl_vel))
        m_rear_right.setVelocity(-velocidad_segura(motor_rr_vel))

if __name__ == '__main__':
    main()