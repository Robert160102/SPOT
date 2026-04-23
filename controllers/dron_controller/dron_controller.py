import math
import os
from controller import Robot

# --- CONFIGURACIÓN DE VUELO ---
ALTURA_VUELO = 2.0       
VELOCIDAD_MAX = 0.1      # Inclinación máxima permitida (para ir despacito)
TOLERANCIA_DIST = 0.5    

# Ganancias del Controlador PID Estabilización
K_VERTICAL_P = 3.0
K_ROLL_P = 50.0
K_PITCH_P = 50.0

# --- SISTEMA DE NAVEGACIÓN Y FRENADO ---
K_POS = 0.02   # Fuerza con la que quiere ir al objetivo
K_VEL = 0.1    # Fuerza con la que frena para no pasarse de largo (Amortiguación)

def cargar_ruta(fichero):
    waypoints = []
    if not os.path.exists(fichero):
        print(f"Error: No se encuentra el fichero {fichero}")
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
    """Función para limitar un valor entre un máximo y un mínimo"""
    return max(-limite, min(limite, valor))

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0  # Tiempo en segundos para calcular velocidades

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

    # 3. Preparar la Ruta
    waypoints = cargar_ruta('ruta.md')
    if len(waypoints) >= 2:
        waypoints.append(calcular_punto_extra(waypoints[-2], waypoints[-1], distancia=1.0))
        print(f"Ruta cargada. Destinos totales: {len(waypoints)} (con inercia).")

    # Función de límite físico ampliado para permitir maniobras sin caerse
    def velocidad_segura(v):
        return max(0.0, min(150.0, v))

    # --- FASE 1: ESPERA DE SEGURIDAD DEL GPS ---
    print("Esperando señal GPS estable...")
    while robot.step(timestep) != -1:
        pos_actual = gps.getValues()
        if not math.isnan(pos_actual[0]):
            print(f"GPS Activado. Posición inicial detectada: {pos_actual}")
            break

    # --- FASE 2: DESPEGUE VERTICAL ESTRICTO ---
    print("Iniciando despegue vertical...")
    while robot.step(timestep) != -1:
        pos = gps.getValues()
        roll, pitch, yaw = imu.getRollPitchYaw()
        gyro_vals = gyro.getValues()
        roll_vel, pitch_vel, yaw_vel = gyro_vals[0], gyro_vals[1], gyro_vals[2]
        
        error_altura = ALTURA_VUELO - pos[1]
        
        if abs(error_altura) < 0.1:
            print("Altura alcanzada. Iniciando navegación de ruta...")
            break
            
        base_thrust = 68.5
        altitude_pid = K_VERTICAL_P * error_altura
        
        # Mantenemos target_roll y target_pitch a 0 para subir completamente rectos
        roll_input = K_ROLL_P * (0.0 - roll) - roll_vel
        pitch_input = K_PITCH_P * (0.0 - pitch) - pitch_vel
        yaw_input = 0.0
        
        motor_fl_vel = base_thrust + altitude_pid - roll_input - pitch_input + yaw_input
        motor_fr_vel = base_thrust + altitude_pid + roll_input - pitch_input - yaw_input
        motor_rl_vel = base_thrust + altitude_pid - roll_input + pitch_input - yaw_input
        motor_rr_vel = base_thrust + altitude_pid + roll_input + pitch_input + yaw_input
        
        m_front_left.setVelocity(-velocidad_segura(motor_fl_vel))
        m_front_right.setVelocity(velocidad_segura(motor_fr_vel))
        m_rear_left.setVelocity(velocidad_segura(motor_rl_vel))
        m_rear_right.setVelocity(-velocidad_segura(motor_rr_vel))

    # --- FASE 3: NAVEGACIÓN PRINCIPAL ---
    current_wp_idx = 0
    modo_hover = False
    last_pos = gps.getValues() # Inicializamos last_pos justo antes de empezar a movernos

    while robot.step(timestep) != -1:
        pos = gps.getValues()
        if math.isnan(pos[0]): continue

        # Calcular velocidad actual para el freno aerodinámico
        vx = (pos[0] - last_pos[0]) / dt
        vz = (pos[2] - last_pos[2]) / dt
        last_pos = pos

        # Lectura de giroscopio e IMU
        roll, pitch, yaw = imu.getRollPitchYaw()
        gyro_vals = gyro.getValues()
        roll_vel, pitch_vel, yaw_vel = gyro_vals[0], gyro_vals[1], gyro_vals[2]

        target = waypoints[current_wp_idx]
        dx = target[0] - pos[0]
        dz = target[2] - pos[2]
        dist_horizontal = math.sqrt(dx**2 + dz**2)

        if dist_horizontal < TOLERANCIA_DIST:
            if current_wp_idx < len(waypoints) - 1:
                current_wp_idx += 1
                print(f"Waypoint {current_wp_idx} alcanzado. Siguiente...")
            elif not modo_hover:
                print("Llegada al punto final. Activando modo estacionario (Hover).")
                modo_hover = True

        # Rotar coordenadas al sistema local del dron
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        # Error de posición local
        err_x_local = dx * cos_yaw + dz * sin_yaw
        err_z_local = -dx * sin_yaw + dz * cos_yaw
        
        # Velocidad local
        vx_local = vx * cos_yaw + vz * sin_yaw
        vz_local = -vx * sin_yaw + vz * cos_yaw

        # FRENADO INTELIGENTE: Combina la distancia (err) con la velocidad (v)
        target_roll = clamp(-err_x_local * K_POS + vx_local * K_VEL, VELOCIDAD_MAX)
        target_pitch = clamp(-err_z_local * K_POS + vz_local * K_VEL, VELOCIDAD_MAX)

        # ---- CONTROLADOR PID ----
        error_altura = ALTURA_VUELO - pos[1]
        base_thrust = 68.5  # Empuje base de flotación
        altitude_pid = K_VERTICAL_P * error_altura

        roll_input = K_ROLL_P * (target_roll - roll) - roll_vel
        pitch_input = K_PITCH_P * (target_pitch - pitch) - pitch_vel
        yaw_input = 0.0

        # MEZCLA DE MOTORES (Fórmula estándar)
        motor_fl_vel = base_thrust + altitude_pid - roll_input - pitch_input + yaw_input
        motor_fr_vel = base_thrust + altitude_pid + roll_input - pitch_input - yaw_input
        motor_rl_vel = base_thrust + altitude_pid - roll_input + pitch_input - yaw_input
        motor_rr_vel = base_thrust + altitude_pid + roll_input + pitch_input + yaw_input

        # Aplicar velocidades respetando la dirección de rotación de cada hélice
        m_front_left.setVelocity(-velocidad_segura(motor_fl_vel))
        m_front_right.setVelocity(velocidad_segura(motor_fr_vel))
        m_rear_left.setVelocity(velocidad_segura(motor_rl_vel))
        m_rear_right.setVelocity(-velocidad_segura(motor_rr_vel))

if __name__ == '__main__':
    main()