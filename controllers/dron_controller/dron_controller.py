"""
Controlador modificado para DJI Mavic 2 PRO.
Lee un archivo 'ruta.md' e ignora encabezados, manteniendo una altura de 2.5m.
"""

from controller import Robot
import sys
import re

try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

def cargar_waypoints_pro(fichero):
    puntos = []
    try:
        with open(fichero, 'r') as f:
            for linea in f:
                linea = linea.strip()
                # Filtramos comentarios, encabezados de Webots y líneas vacías
                if not linea or "Rotation" in linea or "angle" in linea or linea.startswith("#") or "Dron" in linea:
                    continue
                
                # Buscamos las coordenadas numéricas
                coords = re.findall(r"[-+]?\d*\.\d+|\d+", linea)
                if len(coords) >= 3:
                    # Guardamos X y Z para el movimiento horizontal
                    # Ignoramos la Y del archivo (coords[1]) porque fijaremos la altura en el código
                    puntos.append([float(coords[0]), float(coords[2])])
        print(f"✅ Ruta cargada: {len(puntos)} waypoints listos.")
    except Exception as e:
        print(f"❌ Error al leer el archivo '{fichero}': {e}")
    return puntos

class Mavic(Robot):
    # Constantes empíricas para el PID del dron
    K_VERTICAL_THRUST = 68.5  # Empuje para levantar el dron
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # Constante P del PID vertical
    K_ROLL_P = 50.0           # Constante P del PID de alabeo (balanceo)
    K_PITCH_P = 30.0          # Constante P del PID de cabeceo

    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1
    # Precisión de llegada al waypoint en metros
    target_precision = 0.5

    def __init__(self):
        Robot.__init__(self)

        self.time_step = int(self.getBasicTimeStep())

        # Inicializar y habilitar dispositivos
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        # Motores de las hélices
        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        
        # Cámara
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)
        
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        self.current_pose = 6 * [0]  # X, Z, Altitud, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 2.5

    def set_position(self, pos):
        """ Actualiza la posición absoluta del robot """
        self.current_pose = pos

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=False):
        """ Calcula las perturbaciones de cabeceo y guiñada para ir al punto """
        
        if self.target_position[0:2] == [0, 0]:  # Inicialización del primer punto
            self.target_position[0:2] = waypoints[0]
            if verbose_target:
                print("Primer objetivo: ", self.target_position[0:2])

        # Comprobar si hemos llegado al punto con la precisión definida
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):
            self.target_index += 1
            # Si llegamos al final, volvemos a empezar (patrulla infinita)
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
                print("🔄 Ruta completada. Reiniciando patrulla...")
            
            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print("¡Objetivo alcanzado! Nuevo objetivo: ", self.target_position[0:2])

        # Cálculos trigonométricos para orientar el dron hacia el waypoint
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        
        angle_left = self.target_position[2] - self.current_pose[5]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi

        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        pitch_disturbance = clamp(np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose_movement:
            distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
                (self.target_position[1] - self.current_pose[1]) ** 2))
            print("Ángulo restante: {:.4f}, Distancia restante: {:.4f}".format(angle_left, distance_left))
            
        return yaw_disturbance, pitch_disturbance

    def run(self):
        t1 = self.getTime()
        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        # 1. CARGAR LA RUTA
        waypoints = cargar_waypoints_pro("ruta.md")
        
        if not waypoints:
            print("🛑 No se puede iniciar: No hay waypoints válidos.")
            return

        # 2. DEFINIR ALTURA FIJA
        self.target_altitude = 2.5 

        # 3. BUCLE PRINCIPAL DE SIMULACIÓN
        while self.step(self.time_step) != -1:
            
            # Leer sensores
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            
            # EL CORRECCIÓN ESTÁ AQUÍ: La altitud es el tercer valor (Z) en Webots
            x_pos, y_pos, altitude = self.gps.getValues() 
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            
            # Pasamos X e Y como plano horizontal, y la altitud
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            # Solo navegamos hacia los waypoints si ya hemos despegado un poco
            if altitude > 0.5: 
                if self.getTime() - t1 > 0.1:
                    yaw_disturbance, pitch_disturbance = self.move_to_target(waypoints)
                    t1 = self.getTime()

            # --- CONTROLADOR PID Y MEZCLA DE MOTORES ---
            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            
            clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            # Ecuaciones de dinámica de vuelo del cuadricóptero
            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            # Aplicar velocidades a las hélices (fíjate en los signos para contrarrestar el torque)
            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)

# Ejecutar el robot
if __name__ == '__main__':
    robot = Mavic()
    robot.run()