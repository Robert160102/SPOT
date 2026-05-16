"""
DJI Mavic 2 Pro controller with simulated obstacle detection scenario.

This controller behaves like the standard drone controller (waypoint guidance,
home-return safety logic), but it adds a simulated collision-prediction scenario:
shortly after the mission is armed, an obstacle node appears on the planned
route. When the drone gets close to it, the controller:

- Turns on a red emissive sphere above the drone (visual alarm).
- Sends a JSON alert through a dedicated emitter (channel 3) so the supervisor
  can forward it to the web interface.
- Pauses the mission by hovering in place. Because the supervisor projects the
  vehicle target onto the drone position, the car also stops naturally.

After a short timeout (representing the server processing the alarm), the
obstacle node and the red light are removed, an "obstacle_cleared" event is
sent, and both drone and car resume the mission.
"""

from controller import Supervisor
import json
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")


# Development flag used to enable periodic debug information.
dev = True

# Minimum altitude used when the drone is idle or returning to base.
HOME_ALT_MIN = 0.5


# =========================
# OBSTACLE SCENARIO PARAMETERS
# =========================

# World coordinates where the simulated obstacle will appear. The position is
# placed on the main north-south access road, between access_south and
# access_mid, so the drone and the car will encounter it during a normal
# parking mission.
OBSTACLE_XY = (0.0, -40.0)

# XY distance below which the drone considers the obstacle a collision risk.
OBSTACLE_DETECTION_DIST = 6.0

# Seconds the drone hovers with the alarm active before the obstacle is
# considered cleared. This represents the round-trip with the server.
OBSTACLE_PAUSE_DURATION = 3.0

# Seconds after the mission is armed before the obstacle node is spawned. A
# small delay ensures the drone has taken off and started flying before the
# obstacle appears on the route.
OBSTACLE_SPAWN_DELAY = 4.0


def clamp(value, value_min, value_max):
    """
    Limit a value to the provided range.
    """
    return min(max(value, value_min), value_max)


class Mavic(Supervisor):
    """
    Custom Webots controller based on the built-in Mavic drone model.

    The controller reuses the stable low-level motor control structure provided
    by the Webots example and extends it with mission-level behavior, waypoint
    navigation, safety monitoring and a simulated obstacle detection scenario.
    """

    # Basic vertical stabilization constants.
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0

    # Roll and pitch control constants.
    K_ROLL_P = 50.0
    K_PITCH_P = 30.0

    # Maximum yaw correction applied at each control step.
    MAX_YAW_DISTURBANCE = 0.4

    # More negative pitch means stronger forward inclination and higher
    # horizontal cruising speed.
    MAX_PITCH_DISTURBANCE = -3.0

    # Distance-to-pitch gain. A more negative value accelerates the drone
    # faster towards the maximum pitch disturbance.
    PITCH_DIST_GAIN = -0.35

    # Horizontal distance threshold used to consider a waypoint reached.
    target_precision = 1.5

    def __init__(self):
        Supervisor.__init__(self)
        self.time_step = int(self.getBasicTimeStep())

        # =========================
        # SENSOR INITIALIZATION
        # =========================

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
            print("[drone] Warning: receiver device not found. No waypoint missions will be received.")

        # Emitter used to send drone events (e.g., obstacle alerts) back to the
        # supervisor on channel 3.
        self.emitter_supervisor = self.getDevice("emitter_supervisor")
        if self.emitter_supervisor is None:
            print("[drone] Warning: emitter_supervisor device not found. Drone alerts will not be sent.")

        # =========================
        # ACTUATOR INITIALIZATION
        # =========================

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")

        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)

        for motor in [
            self.front_left_motor,
            self.front_right_motor,
            self.rear_left_motor,
            self.rear_right_motor
        ]:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        # =========================
        # MISSION STATE
        # =========================

        self.current_pose = 6 * [0]

        self.waypoints = []
        self.target_index = 0
        self.target_position = [0, 0, 0]
        self.target_altitude = HOME_ALT_MIN

        self.car_node = self.getFromDef("CAR")

        self.home_position = None

        # =========================
        # OBSTACLE SCENARIO STATE
        # =========================

        # State machine for the obstacle scenario:
        #   "absent"   - mission running normally, no obstacle in the world
        #   "present"  - obstacle node exists, waiting for the drone to detect it
        #   "detected" - drone is hovering, alarm active, waiting for timeout
        #   "clearing" - obstacle removed, light pending removal next step
        #   "cleared"  - obstacle and light removed, mission resumed (final state)
        self.obstacle_state = "absent"

        # Time the mission was armed; used to delay the obstacle spawn.
        self.mission_armed_time = None

        # Time the obstacle was detected; used to time the pause.
        self.obstacle_detect_time = None

        # Root children field used to import dynamic VRML nodes.
        self.root_children = self.getRoot().getField("children")

    # =========================
    # OBSTACLE SCENARIO HELPERS
    # =========================

    def _import_node(self, vrml):
        """
        Dynamically insert a VRML node into the Webots world.
        """
        try:
            self.root_children.importMFNodeFromString(-1, vrml)
        except Exception as e:
            print(f"[drone] Warning: importMFNodeFromString failed: {e}")

    def spawn_obstacle(self):
        """
        Create a visible obstacle in the world at OBSTACLE_XY.

        The obstacle is a simple red box ~1.8 m tall. A boundingObject is
        included so Webots treats it as a fully formed Solid; this avoids
        edge-case warnings on some Webots builds.
        """
        x, y = OBSTACLE_XY
        # Height/2 = 0.9 so the box rests on the ground.
        self._import_node(
            f'DEF SIM_OBSTACLE Solid {{ '
            f'translation {x} {y} 0.9 '
            f'name "sim_obstacle" '
            f'children [ Shape {{ '
            f'appearance Appearance {{ material Material {{ '
            f'diffuseColor 0.9 0.1 0.1 emissiveColor 0.4 0.0 0.0 }} }} '
            f'geometry Box {{ size 0.6 0.6 1.8 }} }} ] '
            f'boundingObject Box {{ size 0.6 0.6 1.8 }} }}'
        )
        print(f"[drone] Simulated obstacle spawned at ({x:.2f}, {y:.2f}).")

    def remove_node_by_def(self, def_name):
        """
        Remove a previously imported node looked up by DEF name.

        Wrapped in try/except because operating on dynamically imported nodes
        through the Supervisor API can occasionally raise low-level errors
        depending on the Webots build.
        """
        try:
            n = self.getFromDef(def_name)
            if n is not None:
                n.remove()
        except Exception as e:
            print(f"[drone] Warning: failed to remove {def_name}: {e}")

    def spawn_alert_light(self, x, y, z):
        """
        Create a red emissive sphere above the drone position at the moment of
        detection. The light is NOT updated each step: the drone is hovering
        during the alarm, so a fixed beacon at the detection position is
        visually equivalent to one that follows it. Removing the per-step
        setSFVec3f calls eliminates a class of crashes seen when modifying
        a dynamically imported node every step.
        """
        self._import_node(
            f'DEF DRONE_ALERT_LIGHT Solid {{ '
            f'translation {x} {y} {z + 0.6} '
            f'name "drone_alert_light" '
            f'children [ Shape {{ '
            f'appearance Appearance {{ material Material {{ '
            f'diffuseColor 1 0 0 emissiveColor 1 0 0 }} }} '
            f'geometry Sphere {{ radius 0.35 }} }} ] }}'
        )

    def send_event(self, payload):
        """
        Send a JSON event to the supervisor through the dedicated emitter.
        """
        if self.emitter_supervisor is None:
            return

        try:
            self.emitter_supervisor.send(json.dumps(payload).encode('utf-8'))
        except Exception as e:
            print(f"[drone] Failed to send event: {e}")

    def set_position(self, pos):
        self.current_pose = pos

    def consume_messages(self):
        """
        Read and parse pending waypoint missions from the receiver queue.
        """
        if self.receiver is None:
            return

        while self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            self.receiver.nextPacket()

            try:
                payload = json.loads(data)
            except Exception:
                print(f"[drone] Invalid receiver payload: {data[:80]}")
                continue

            wps = payload.get("waypoints")

            if not isinstance(wps, list) or len(wps) == 0:
                continue

            self.waypoints = [list(w) for w in wps]
            self.target_index = 0
            self.target_position = [
                self.waypoints[0][0],
                self.waypoints[0][1],
                0
            ]
            self.target_altitude = self.waypoints[0][2]

            print(f"[drone] Received {len(self.waypoints)} waypoints. First waypoint: {self.waypoints[0]}")

    def move_to_target(self):
        """
        Compute yaw and pitch disturbances required to move towards the target.
        """
        if not self.waypoints:
            return 0.0, 0.0

        target_x, target_y = self.target_position[0], self.target_position[1]
        dist_x = target_x - self.current_pose[0]
        dist_y = target_y - self.current_pose[1]
        distance_left = np.sqrt(dist_x ** 2 + dist_y ** 2)

        is_last = self.target_index >= len(self.waypoints) - 1
        precision = self.target_precision

        if distance_left < precision:
            if is_last:
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

        yaw_disturbance = clamp(
            angle_left,
            -self.MAX_YAW_DISTURBANCE,
            self.MAX_YAW_DISTURBANCE
        )

        if abs(angle_left) < 0.5:
            pitch_disturbance = clamp(
                distance_left * self.PITCH_DIST_GAIN,
                self.MAX_PITCH_DISTURBANCE,
                0
            )
        else:
            pitch_disturbance = 0

        return yaw_disturbance, pitch_disturbance

    def stop_motors(self):
        for motor in (
            self.front_left_motor,
            self.front_right_motor,
            self.rear_left_motor,
            self.rear_right_motor
        ):
            motor.setVelocity(0)

    def update_obstacle_scenario(self, x_pos, y_pos, altitude):
        """
        Drive the obstacle scenario state machine.

        Returns True while the drone must hover in place because of an active
        obstacle alarm. The main loop uses this to override the normal
        navigation outputs.
        """
        now = self.getTime()

        # =========================
        # ABSENT -> PRESENT
        # =========================
        if (
            self.obstacle_state == "absent"
            and self.mission_armed_time is not None
            and now - self.mission_armed_time >= OBSTACLE_SPAWN_DELAY
        ):
            self.spawn_obstacle()
            self.obstacle_state = "present"

        # =========================
        # PRESENT -> DETECTED
        # =========================
        if self.obstacle_state == "present":
            ox, oy = OBSTACLE_XY
            d = np.sqrt((x_pos - ox) ** 2 + (y_pos - oy) ** 2)

            if d < OBSTACLE_DETECTION_DIST:
                print(
                    f"[drone] Obstacle detected at {d:.2f} m. "
                    f"Activating alarm and pausing mission."
                )

                self.spawn_alert_light(x_pos, y_pos, altitude)
                self.send_event({
                    "type": "obstacle_detected",
                    "position": [ox, oy],
                    "drone_position": [x_pos, y_pos, altitude],
                    "distance": float(d),
                    "time": now,
                })

                self.obstacle_state = "detected"
                self.obstacle_detect_time = now

                # Lock the navigation target to the current pose so the drone
                # has no inertia-driven drift target to chase after the pause.
                self.target_position[0] = x_pos
                self.target_position[1] = y_pos
                self.target_altitude = altitude

        # =========================
        # DETECTED (hovering) -> CLEARING
        # =========================
        # In the CLEARING step we ONLY remove the obstacle. The alarm light is
        # removed on the next step (CLEARING -> CLEARED) and the navigation
        # target is restored on the step after that. Spreading these scene
        # graph mutations across consecutive steps avoids Webots crashes that
        # appear when several imported nodes are removed in the same tick.
        if self.obstacle_state == "detected":
            if now - self.obstacle_detect_time >= OBSTACLE_PAUSE_DURATION:
                print("[drone] Server acknowledged alarm. Removing obstacle.")
                self.remove_node_by_def("SIM_OBSTACLE")
                self.obstacle_state = "clearing"

        # =========================
        # CLEARING -> CLEARED
        # =========================
        elif self.obstacle_state == "clearing":
            print("[drone] Removing alarm light and resuming mission.")
            self.remove_node_by_def("DRONE_ALERT_LIGHT")
            self.send_event({
                "type": "obstacle_cleared",
                "time": now,
            })

            self.obstacle_state = "cleared"

            # Restore the original waypoint as navigation target so the drone
            # resumes flying along the planned route.
            if self.waypoints and self.target_index < len(self.waypoints):
                wp = self.waypoints[self.target_index]
                self.target_position[0] = wp[0]
                self.target_position[1] = wp[1]
                self.target_altitude = wp[2]

        # The drone must hover while the alarm is active OR while we are
        # removing nodes. It only returns to normal flight in CLEARED.
        return self.obstacle_state in ("detected", "clearing")

    def run(self):
        """
        Main drone control loop.
        """
        self.target_position = [0.0, 0.0, 0.0]
        self.target_altitude = HOME_ALT_MIN

        yaw_disturbance = 0.0
        pitch_disturbance = 0.0

        mission_armed = False

        last_log = 0.0
        mission_aborted = False

        t1 = self.getTime()

        while self.step(self.time_step) != -1:
            had_no_waypoints = not self.waypoints

            # =========================
            # A. MISSION RECEPTION
            # =========================

            if mission_aborted:
                while self.receiver is not None and self.receiver.getQueueLength() > 0:
                    self.receiver.nextPacket()
            else:
                self.consume_messages()

            if had_no_waypoints and self.waypoints:
                mission_armed = True
                self.mission_armed_time = self.getTime()
                print(f"[drone] Mission armed with {len(self.waypoints)} waypoints.")

            # =========================
            # B. CURRENT DRONE STATE
            # =========================

            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()

            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            if self.home_position is None and not np.isnan(x_pos):
                self.home_position = [x_pos, y_pos, HOME_ALT_MIN]

            # =========================
            # C. SAFETY MONITORING (vehicle lost)
            # =========================

            if self.waypoints and self.car_node and mission_armed and not mission_aborted:
                car_position = self.car_node.getPosition()
                car_distance = np.sqrt(
                    (car_position[0] - x_pos) ** 2 +
                    (car_position[1] - y_pos) ** 2
                )

                MAX_CAR_DISTANCE = 25.0

                if car_distance > MAX_CAR_DISTANCE:
                    print(f"[drone] Alert: vehicle lost at {car_distance:.1f} m. Aborting mission.")
                    mission_aborted = True

                    return_wp = [
                        self.home_position[0],
                        self.home_position[1],
                        self.target_altitude
                    ]
                    landing_wp = [
                        self.home_position[0],
                        self.home_position[1],
                        HOME_ALT_MIN
                    ]

                    self.waypoints = [return_wp, landing_wp]
                    self.target_index = 0

                    self.target_position[0] = return_wp[0]
                    self.target_position[1] = return_wp[1]
                    self.target_altitude = return_wp[2]

            # =========================
            # D. OBSTACLE SCENARIO
            # =========================

            obstacle_hold = False
            if mission_armed and not mission_aborted:
                obstacle_hold = self.update_obstacle_scenario(x_pos, y_pos, altitude)

            # =========================
            # E. IDLE / MISSION CONTROL
            # =========================

            if not mission_armed:
                self.stop_motors()
                continue

            if obstacle_hold:
                # Force the drone to hover in place while the alarm is active.
                yaw_disturbance = 0.0
                pitch_disturbance = 0.0
            elif self.waypoints:
                if altitude > self.target_altitude - 0.3:
                    if self.getTime() - t1 > 0.1:
                        yaw_disturbance, pitch_disturbance = self.move_to_target()
                        t1 = self.getTime()
                else:
                    yaw_disturbance = 0.0
                    pitch_disturbance = 0.0
            else:
                self.target_position[0] = x_pos
                self.target_position[1] = y_pos
                yaw_disturbance = 0.0
                pitch_disturbance = 0.0

            # =========================
            # F. PERIODIC DEBUG LOGGING
            # =========================

            now = self.getTime()

            if now - last_log > 1.0:
                if obstacle_hold:
                    print(
                        f"[drone] HOVER (alarm active) "
                        f"pos=({x_pos:.1f},{y_pos:.1f},{altitude:.1f}) "
                        f"obstacle_state={self.obstacle_state}"
                    )
                elif self.waypoints:
                    tx, ty, tz = (
                        self.target_position[0],
                        self.target_position[1],
                        self.target_altitude
                    )

                    dx_t = tx - x_pos
                    dy_t = ty - y_pos
                    distance_to_target = (dx_t ** 2 + dy_t ** 2) ** 0.5
                    target_angle = np.arctan2(dy_t, dx_t)
                    angle_left = (target_angle - yaw + np.pi) % (2 * np.pi) - np.pi

                    if dev:
                        print(
                            f"[drone] wp {self.target_index + 1}/{len(self.waypoints)} "
                            f"pos=({x_pos:.1f},{y_pos:.1f},{altitude:.1f}) "
                            f"target=({tx:.1f},{ty:.1f},{tz:.1f}) "
                            f"dxy={distance_to_target:.1f} "
                            f"yaw={yaw:.2f} "
                            f"target_angle={target_angle:.2f} "
                            f"angle_left={angle_left:.2f} "
                            f"yawD={yaw_disturbance:.2f} "
                            f"pitchD={pitch_disturbance:.2f} "
                            f"obs={self.obstacle_state}"
                        )
                else:
                    print(f"[drone] Hovering at ({x_pos:.1f},{y_pos:.1f},{altitude:.1f})")

                last_log = now

            # =========================
            # G. LOW-LEVEL MOTOR CONTROL
            # =========================

            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration
            pitch_input = (
                self.K_PITCH_P * clamp(pitch, -1, 1) +
                pitch_acceleration +
                pitch_disturbance
            )
            yaw_input = yaw_disturbance

            clamped_diff_alt = clamp(
                self.target_altitude - altitude + self.K_VERTICAL_OFFSET,
                -1,
                1
            )
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
