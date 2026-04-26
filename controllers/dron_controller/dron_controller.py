"""
Controlador DJI Mavic 2 PRO.
Recibe lista de waypoints del parking_supervisor por Receiver canal 2.
Sin waypoints: hover en posicion actual.
"""

from controller import Robot
import json
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")


dev = True  # True para debug, False no debug (menos prints, etc)

HOME_ALT_MIN = 0.5


def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


class Mavic(Robot):
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0
    K_ROLL_P = 50.0
    K_PITCH_P = 30.0

    MAX_YAW_DISTURBANCE = 0.4
    # Pitch mas negativo => mayor inclinacion hacia adelante => mas velocidad de crucero.
    # -3.0 lo lleva a una velocidad horizontal claramente superior a la del coche.
    MAX_PITCH_DISTURBANCE = -3.0
    # Ganancia distancia->pitch (mas negativa => acelera antes a la velocidad max)
    PITCH_DIST_GAIN = -0.35

    target_precision = 1.5

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())

        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.receiver = self.getDevice("receiver")
        if self.receiver is not None:
            self.receiver.enable(self.time_step)
        else:
            print("AVISO: el dron no tiene Receiver. No recibira waypoints.")

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)

        for motor in [self.front_left_motor, self.front_right_motor,
                      self.rear_left_motor, self.rear_right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        self.current_pose = 6 * [0]   # x, y, z, roll, pitch, yaw
        self.waypoints = []
        self.target_index = 0
        self.target_position = [0, 0, 0]
        self.target_altitude = 0.5    # arranca a baja altura

    def set_position(self, pos):
        self.current_pose = pos

    def consume_messages(self):
        if self.receiver is None:
            return
        while self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            self.receiver.nextPacket()
            try:
                payload = json.loads(data)
            except Exception:
                print(f"Mensaje invalido: {data[:80]}")
                continue
            wps = payload.get("waypoints")
            if not isinstance(wps, list) or len(wps) == 0:
                continue
            self.waypoints = [list(w) for w in wps]
            self.target_index = 0
            self.target_position = [self.waypoints[0][0], self.waypoints[0][1], 0]
            self.target_altitude = self.waypoints[0][2]
            print(f"Recibidos {len(self.waypoints)} waypoints. Primero: {self.waypoints[0]}")

    def move_to_target(self):
        """Devuelve (yaw_disturbance, pitch_disturbance). Avanza indice si llega."""
        if not self.waypoints:
            return 0.0, 0.0

        target_x, target_y = self.target_position[0], self.target_position[1]
        dist_x = target_x - self.current_pose[0]
        dist_y = target_y - self.current_pose[1]
        distance_left = np.sqrt(dist_x ** 2 + dist_y ** 2)

        is_last = self.target_index >= len(self.waypoints) - 1
        # Para waypoints finales (descenso/ascenso) usa precision menor en xy
        precision = self.target_precision

        if distance_left < precision:
            if is_last:
                # Ya en el ultimo: mantener altura objetivo, no avanzar
                return 0.0, 0.0
            self.target_index += 1
            wp = self.waypoints[self.target_index]
            self.target_position[0] = wp[0]
            self.target_position[1] = wp[1]
            self.target_altitude = wp[2]
            target_x, target_y = wp[0], wp[1]
            dist_x = target_x - self.current_pose[0]
            dist_y = target_y - self.current_pose[1]
            distance_left = np.sqrt(dist_x ** 2 + dist_y ** 2)

        target_angle = np.arctan2(dist_y, dist_x)
        angle_left = target_angle - self.current_pose[5]
        angle_left = (angle_left + np.pi) % (2 * np.pi) - np.pi

        yaw_disturbance = clamp(angle_left, -self.MAX_YAW_DISTURBANCE, self.MAX_YAW_DISTURBANCE)
        if abs(angle_left) < 0.5:
            pitch_disturbance = clamp(distance_left * self.PITCH_DIST_GAIN,
                                      self.MAX_PITCH_DISTURBANCE, 0)
        else:
            pitch_disturbance = 0
        return yaw_disturbance, pitch_disturbance

    def stop_motors(self):
        for m in (self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor):
            m.setVelocity(0)

    def run(self):
        self.target_position = [0.0, 0.0, 0.0]
        self.target_altitude = HOME_ALT_MIN
        yaw_disturbance = 0.0
        pitch_disturbance = 0.0
        mission_armed = False  # True una vez recibido primer waypoint
        last_log = 0.0

        t1 = self.getTime()
        while self.step(self.time_step) != -1:
            had_no_wps = not self.waypoints
            self.consume_messages()
            if had_no_wps and self.waypoints:
                mission_armed = True
                print(f"[dron] mision armada con {len(self.waypoints)} waypoints")

            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            # Sin mision aun -> motores apagados (en suelo)
            if not mission_armed:
                self.stop_motors()
                continue

            if self.waypoints:
                if altitude > self.target_altitude - 0.3:
                    if self.getTime() - t1 > 0.1:
                        yaw_disturbance, pitch_disturbance = self.move_to_target()
                        t1 = self.getTime()
                else:
                    yaw_disturbance = 0.0
                    pitch_disturbance = 0.0
            else:
                # Sin waypoints pero ya armado -> hover en sitio actual
                self.target_position[0] = x_pos
                self.target_position[1] = y_pos
                yaw_disturbance = 0.0
                pitch_disturbance = 0.0

            # Log periodico del estado
            now = self.getTime()
            if now - last_log > 1.0:
                if self.waypoints:
                    tx, ty, tz = (self.target_position[0], self.target_position[1], self.target_altitude)
                    dx_t, dy_t = tx - x_pos, ty - y_pos
                    d = (dx_t ** 2 + dy_t ** 2) ** 0.5
                    tgt_ang = np.arctan2(dy_t, dx_t)
                    al = (tgt_ang - yaw + np.pi) % (2 * np.pi) - np.pi
                    if dev:
                        print(f"[dron] wp {self.target_index+1}/{len(self.waypoints)} "
                            f"pos=({x_pos:.1f},{y_pos:.1f},{altitude:.1f}) "
                            f"target=({tx:.1f},{ty:.1f},{tz:.1f}) dxy={d:.1f} "
                            f"yaw={yaw:.2f} tgt_ang={tgt_ang:.2f} angle_left={al:.2f} "
                            f"yawD={yaw_disturbance:.2f} pitchD={pitch_disturbance:.2f}")
                else:
                    print(f"[dron] hover en ({x_pos:.1f},{y_pos:.1f},{altitude:.1f})")
                last_log = now

            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_diff_alt = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_diff_alt, 3.0)

            m1 = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            m2 = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            m3 = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            m4 = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            self.front_left_motor.setVelocity(m1)
            self.front_right_motor.setVelocity(-m2)
            self.rear_left_motor.setVelocity(-m3)
            self.rear_right_motor.setVelocity(m4)


if __name__ == '__main__':
    robot = Mavic()
    robot.run()
