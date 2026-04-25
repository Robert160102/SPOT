"""
Controlador modificado para DJI Mavic 2 PRO.
Lee un archivo 'ruta.md' e ignora encabezados, manteniendo una altura de 2.5m.
Envía su posición en el plano XY al coche seguidor.
"""

from controller import Robot
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

WAYPOINTS = [
    [-2.17, -70.27, 2.5], [-2.17, -68.77, 2.5], [-2.17, -67.27, 2.5], [-2.17, -65.77, 2.5],
    [-2.17, -64.27, 2.5], [-2.17, -62.77, 2.5], [-2.17, -61.27, 2.5], [-2.17, -59.77, 2.5],
    [-2.17, -58.27, 2.5], [-2.17, -56.77, 2.5], [-2.17, -55.27, 2.5], [-2.17, -53.77, 2.5],
    [-2.17, -52.27, 2.5], [-2.17, -50.77, 2.5], [-2.17, -49.27, 2.5], [-2.17, -47.77, 2.5],
    [-2.17, -46.27, 2.5], [-2.17, -44.77, 2.5], [-2.17, -43.27, 2.5], [-2.17, -41.77, 2.5],
    [-2.17, -40.27, 2.5], [-2.17, -38.77, 2.5], [-2.17, -37.27, 2.5], [-2.17, -35.77, 2.5],
    [-2.17, -34.27, 2.5], [-2.17, -32.77, 2.5], [-2.17, -31.27, 2.5], [-2.17, -29.77, 2.5],
    [-2.17, -28.27, 2.5], [-2.17, -26.77, 2.5], [-2.17, -25.27, 2.5], [-2.17, -23.77, 2.5],
    [-2.17, -22.27, 2.5], [-2.17, -20.77, 2.5], [-2.17, -19.27, 2.5], [-2.17, -17.77, 2.5],
    [-2.17, -16.27, 2.5], [-2.17, -14.77, 2.5], [-2.17, -13.27, 2.5], [-2.17, -11.77, 2.5],
    [-2.17, -10.27, 2.5], [-2.17,  -8.77, 2.5], [-2.17,  -7.27, 2.5], [-2.17,  -5.77, 2.5],
    [-2.17,  -4.27, 2.5], [-2.17,  -2.77, 2.5], [-2.17,  -2.63, 2.5], [-0.67,  -2.63, 2.5],
    [ 0.83,  -2.63, 2.5], [ 2.33,  -2.63, 2.5], [ 3.83,  -2.63, 2.5], [ 5.33,  -2.63, 2.5],
    [ 6.83,  -2.63, 2.5], [ 8.33,  -2.63, 2.5], [ 9.83,  -2.63, 2.5], [11.33,  -2.63, 2.5],
    [12.83,  -2.63, 2.5], [14.33,  -2.63, 2.5], [15.83,  -2.63, 2.5], [17.33,  -2.63, 2.5],
    [18.83,  -2.63, 2.5], [20.33,  -2.63, 2.5], [21.83,  -2.63, 2.5], [23.33,  -2.63, 2.5],
    [24.83,  -2.63, 2.5], [26.33,  -2.63, 2.5], [27.83,  -2.63, 2.5], [29.33,  -2.63, 2.5],
    [30.83,  -2.63, 2.5], [32.33,  -2.63, 2.5], [33.83,  -2.63, 2.5], [35.33,  -2.63, 2.5],
    [36.83,  -2.63, 2.5], [38.33,  -2.63, 2.5], [39.83,  -2.63, 2.5], [41.33,  -2.63, 2.5],
    [42.83,  -2.63, 2.5], [44.33,  -2.63, 2.5], [45.83,  -2.63, 2.5], [47.33,  -2.63, 2.5],
    [48.83,  -2.63, 2.5], [50.33,  -2.63, 2.5], [51.83,  -2.63, 2.5], [53.33,  -2.63, 2.5],
    [54.83,  -2.63, 2.5], [56.33,  -2.63, 2.5], [57.83,  -2.63, 2.5], [59.33,  -2.63, 2.5],
    [60.83,  -2.63, 2.5], [62.33,  -2.63, 2.5], [63.83,  -2.63, 2.5], [65.33,  -2.63, 2.5],
    [66.83,  -2.63, 2.5], [67.11,  -2.63, 2.5], [67.11,  -1.13, 2.5], [67.11,   0.37, 2.5],
    [67.11,   1.87, 2.5], [67.11,   3.37, 2.5], [67.11,   4.87, 2.5], [67.11,   6.37, 2.5],
    [67.11,   7.87, 2.5], [67.11,   9.37, 2.5], [67.11,  10.87, 2.5], [67.11,  12.37, 2.5],
    [67.11,  13.87, 2.5], [67.11,  15.37, 2.5], [67.11,  16.87, 2.5], [67.11,  18.37, 2.5],
    [67.11,  19.87, 2.5], [67.11,  21.37, 2.5], [67.11,  22.87, 2.5], [67.11,  24.37, 2.5],
    [67.11,  25.87, 2.5], [67.11,  27.37, 2.5], [67.11,  28.87, 2.5], [67.11,  30.37, 2.5],
    [67.11,  31.87, 2.5], [67.11,  33.37, 2.5], [67.11,  34.87, 2.5], [67.11,  36.37, 2.5],
    [67.11,  37.87, 2.5], [67.11,  39.37, 2.5], [67.11,  40.87, 2.5], [67.11,  41.23, 2.5],
    [65.61,  41.23, 2.5], [64.11,  41.23, 2.5], [62.61,  41.23, 2.5], [61.11,  41.23, 2.5],
    [59.61,  41.23, 2.5], [58.11,  41.23, 2.5], [56.61,  41.23, 2.5], [55.11,  41.23, 2.5],
    [54.7,   41.23, 2.5], [54.7,   39.73, 2.5], [54.7,   38.23, 2.5], [54.7,   36.73, 2.5],
    [54.7,   35.23, 2.5], [54.7,   33.74, 2.5], [54.7,   33.74, 4.0], [54.7,   33.74, 5.0],
]

class Mavic(Robot):
    # Constantes empíricas para el PID del dron ajustadas
    K_VERTICAL_THRUST = 68.5  
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # Un poco más fuerte para mantener bien la altura
    K_ROLL_P = 50.0           
    K_PITCH_P = 30.0          

    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1.5 # Empuje hacia adelante

    # Precisión de llegada al waypoint en metros
    target_precision = 0.8

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
        
        # EL EMISOR: Es vital que esté añadido en el árbol de nodos de Webots
        self.emitter = self.getDevice("emitter")

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

        self.current_pose = 6 * [0]  # X, Y, Z(Altitud), roll, pitch, yaw
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 2.5

    def set_position(self, pos):
        """ Actualiza la posición absoluta del robot """
        self.current_pose = pos

    def move_to_target(self, waypoints):
        """ Calcula las perturbaciones de cabeceo y guiñada para ir al punto """
        
        if self.target_position[0:2] == [0, 0]:  # Inicialización del primer punto
            self.target_position[0:2] = waypoints[0][:2]
            self.target_altitude = waypoints[0][2]

        target_x = self.target_position[0]
        target_y = self.target_position[1]

        # Calcular distancia en el plano horizontal (XY)
        dist_x = target_x - self.current_pose[0]
        dist_y = target_y - self.current_pose[1]
        distance_left = np.sqrt(dist_x**2 + dist_y**2)

        # Comprobar si hemos llegado al punto
        if distance_left < self.target_precision:
            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
                print("🔄 Ruta completada. Reiniciando patrulla...")

            self.target_position[0:2] = waypoints[self.target_index][:2]
            self.target_altitude = waypoints[self.target_index][2]
            
            # Recalcular distancias para el nuevo punto y evitar saltos bruscos
            target_x = self.target_position[0]
            target_y = self.target_position[1]
            dist_x = target_x - self.current_pose[0]
            dist_y = target_y - self.current_pose[1]
            distance_left = np.sqrt(dist_x**2 + dist_y**2)

        # Ángulo objetivo en el plano XY
        target_angle = np.arctan2(dist_y, dist_x)
        
        # Calcular cuánto nos queda por girar
        angle_left = target_angle - self.current_pose[5]
        
        # Normalizar el ángulo entre -PI y PI
        angle_left = (angle_left + np.pi) % (2 * np.pi) - np.pi

        # GUIÑADA (Yaw): Girar hacia el objetivo
        yaw_disturbance = clamp(angle_left, -self.MAX_YAW_DISTURBANCE, self.MAX_YAW_DISTURBANCE)
        
        # CABECEO (Pitch): Avanzar solo si estamos más o menos orientados
        if abs(angle_left) < 0.5:
            # Ahora es proporcional a la distancia, no logarítmico (¡adiós efecto delfín!)
            pitch_disturbance = clamp(distance_left * -0.2, self.MAX_PITCH_DISTURBANCE, 0)
        else:
            pitch_disturbance = 0

        return yaw_disturbance, pitch_disturbance

    def run(self):
        t1 = self.getTime()
        waypoints = WAYPOINTS
        print(f"✅ Ruta cargada: {len(waypoints)} waypoints listos.")

        self.target_altitude = waypoints[0][2]

        while self.step(self.time_step) != -1:
            
            # 1. LEER SENSORES
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            
            # Ejes ENU: X (Este), Y (Norte), Z (Arriba/Altitud)
            x_pos, y_pos, altitude = self.gps.getValues() 
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            # 2. ENVIAR DATOS AL COCHE
            if self.emitter:
                # Enviamos solo X e Y (el plano del suelo)
                mensaje = f"{x_pos},{y_pos}".encode('utf-8')
                self.emitter.send(mensaje)

            # 3. NAVEGACIÓN
            if altitude > self.target_altitude - 0.2: 
                if self.getTime() - t1 > 0.1:
                    yaw_disturbance, pitch_disturbance = self.move_to_target(waypoints)
                    t1 = self.getTime()
            else:
                yaw_disturbance = 0
                pitch_disturbance = 0

            # 4. CONTROLADOR PID
            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            
            # NOTA: Si el dron rota sin control, cambia el signo a yaw_disturbance positivo.
            yaw_input = -yaw_disturbance 
            
            clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            # 5. MEZCLA DE MOTORES
            m1 = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            m2 = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            m3 = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            m4 = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            self.front_left_motor.setVelocity(m1)
            self.front_right_motor.setVelocity(-m2)
            self.rear_left_motor.setVelocity(-m3)
            self.rear_right_motor.setVelocity(m4)

# Ejecutar el robot
if __name__ == '__main__':
    robot = Mavic()
    robot.run()