"""
DJI Mavic 2 Pro controller for the SPOT Webots simulation.

This controller receives a list of waypoints from the parking supervisor through
a Webots Receiver node on channel 2. Once the mission is armed, the drone follows
the waypoint sequence and acts as the leading agent for the guided parking demo.

If no waypoints are available, the drone remains inactive on the ground. During
an active mission, the controller monitors the vehicle in two ways:
  - distance check: if the car falls more than MAX_CAR_DISTANCE m behind, the
    mission is aborted (vehicle lost / car stopped).
  - overtake check: if the car projects ahead of the drone along the current
    travel direction for more than OVERTAKE_DEBOUNCE seconds, the mission is
    also aborted (the car is no longer following the drone).
In both cases the drone emits a "mission_aborted" event on channel 3 so the
supervisor can free the reserved spot, and it returns to its home position.
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

# Maximum allowed horizontal distance between drone and vehicle during a
# mission. If the car falls further behind than this, the mission is aborted.
MAX_CAR_DISTANCE = 25.0

# Overtake detection: if the projection of the car position on the drone
# forward direction is greater than this value (in meters), the car is
# considered to be ahead of the drone along the planned route.
OVERTAKE_AHEAD_THRESHOLD = 3.0

# Number of seconds the overtake condition must hold before triggering the
# abort. This debounce avoids false positives at start-up and during turns.
OVERTAKE_DEBOUNCE = 1.0

# Horizontal distance (meters) at which the drone considers it has reached
# the car for the first time during this mission. Until this happens, the
# overtake detection is disabled. This is needed because the car starts at
# access_south (already on the route) while the drone starts at home behind
# it, so at mission arming the car geometrically projects ahead of the drone.
# Once the drone catches up to the car this close, leader-follower behavior
# becomes meaningful and overtake detection can start.
OVERTAKE_ARM_DISTANCE = 3.0

# Solid model string set on the obstacle car's recognition tag. The drone
# uses this to filter the recognized objects returned by the camera and
# distinguish the obstacle from any other vehicle in view. Worlds without
# an obstacle car never see this string, so the scenario stays inactive.
OBSTACLE_MODEL = "obstaculo"


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
    navigation and safety monitoring.
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

        # Camera is enabled for consistency with the Mavic model, even if the
        # current controller mainly relies on GPS and IMU data.
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)

        # Try to enable Recognition on the camera. Only worlds that include a
        # Recognition node inside the Camera will actually return objects.
        # When the world has no Recognition node this call fails silently and
        # the obstacle scenario stays inactive.
        try:
            self.camera.recognitionEnable(self.time_step)
            self.recognition_enabled = True
        except Exception:
            self.recognition_enabled = False

        # IMU provides roll, pitch and yaw.
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)

        # GPS provides the drone position inside the Webots world.
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)

        # Gyro provides angular acceleration used for stabilization.
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        # Receiver node used to obtain waypoint missions from the supervisor.
        # In this implementation, the supervisor sends complete waypoint lists
        # as JSON payloads.
        self.receiver = self.getDevice("receiver")
        if self.receiver is not None:
            self.receiver.enable(self.time_step)
        else:
            print("[drone] Warning: receiver device not found. No waypoint missions will be received.")

        # Emitter used to send drone events (e.g., mission_aborted) back to the
        # supervisor on channel 3.
        self.emitter_supervisor = self.getDevice("emitter_supervisor")
        if self.emitter_supervisor is None:
            print("[drone] Warning: emitter_supervisor device not found. Drone events will not be sent.")

        # Optional emitter used to trigger the obstacle car (channel 4) when
        # the collision-prevention scenario is set up in the current world.
        # When the emitter is not present the obstacle car simply never gets
        # told to move; the scenario is effectively disabled for that world.
        self.emitter_obstaculo = self.getDevice("emitter_obstaculo")

        # =========================
        # ACTUATOR INITIALIZATION
        # =========================

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")

        # Camera pitch is fixed to provide a stable forward/downward view.
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)

        # Motors are set to velocity control mode.
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

        # Current drone pose: x, y, z, roll, pitch, yaw.
        self.current_pose = 6 * [0]

        # Waypoints received from the supervisor.
        self.waypoints = []
        self.target_index = 0
        self.target_position = [0, 0, 0]
        self.target_altitude = HOME_ALT_MIN

        # Reference to the simulated vehicle node. The Webots world must define
        # the car with DEF name "CAR" for this lookup to work.
        self.car_node = self.getFromDef("CAR")

        # Home position is stored once the simulation provides a valid GPS value.
        self.home_position = None

        # Timestamp at which the overtake condition started being true. Set to
        # None when the car is not currently projecting ahead of the drone.
        self.overtake_start_time = None

        # Latch flag. The overtake detection only runs after the drone has
        # been ahead of the car at least once during this mission.
        self.overtake_detection_armed = False

        # =========================
        # OBSTACLE SCENARIO STATE
        # =========================

        # State machine for the optional collision-prevention scenario:
        #   "armed"   - mission running, scanning for the obstacle.
        #   "visible" - obstacle in view, drone hovering, alarm active.
        #   "cleared" - obstacle no longer in view, mission resumed.
        # The scenario only progresses past "armed" if the world actually has
        # a recognizable obstacle car. Otherwise it stays in "armed" forever
        # without affecting normal flight.
        self.obstacle_state = "armed"

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

    def trigger_obstacle_car(self, message):
        """
        Send a plain string command to the obstacle car on channel 4.

        Does nothing in worlds without an obstacle emitter.
        """
        if self.emitter_obstaculo is None:
            return

        try:
            self.emitter_obstaculo.send(message.encode('utf-8'))
        except Exception as e:
            print(f"[drone] Failed to send obstacle command: {e}")

    def obstacle_in_view(self):
        """
        Return True if camera recognition currently reports any object whose
        model field equals OBSTACLE_MODEL. Returns False when recognition is
        not enabled in the current world.
        """
        if not self.recognition_enabled:
            return False

        try:
            objects = self.camera.getRecognitionObjects()
        except Exception:
            return False

        for obj in objects:
            model = obj.getModel() if hasattr(obj, "getModel") else getattr(obj, "model", "")
            if isinstance(model, bytes):
                model = model.decode("utf-8", errors="ignore")
            if model == OBSTACLE_MODEL:
                return True

        return False

    def car_projection_ahead(self, x_pos, y_pos):
        """
        Return the projection (in meters) of the car position on the drone
        forward direction, along the path towards the next waypoint.

        Positive values mean the car is ahead of the drone. Negative values
        mean the car is behind. Returns None when the projection cannot be
        evaluated (no waypoints, last waypoint reached, or degenerate vector).
        """
        if self.car_node is None or not self.waypoints:
            return None

        if self.target_index >= len(self.waypoints):
            return None

        target_x = self.target_position[0]
        target_y = self.target_position[1]

        forward_x = target_x - x_pos
        forward_y = target_y - y_pos
        forward_norm = np.sqrt(forward_x ** 2 + forward_y ** 2)

        if forward_norm < 1e-3:
            return None

        car_position = self.car_node.getPosition()
        rel_x = car_position[0] - x_pos
        rel_y = car_position[1] - y_pos

        return (rel_x * forward_x + rel_y * forward_y) / forward_norm

    def set_position(self, pos):
        """
        Store the current drone pose.
        """
        self.current_pose = pos

    def consume_messages(self):
        """
        Read and parse pending waypoint missions from the receiver queue.

        Expected payload format:
            {
                "waypoints": [[x1, y1, z1], [x2, y2, z2], ...]
            }

        When a valid mission is received, the first waypoint becomes the
        immediate navigation target.
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

        The method also advances to the next waypoint once the current waypoint
        is reached according to the configured target precision.
        """
        if not self.waypoints:
            return 0.0, 0.0

        target_x, target_y = self.target_position[0], self.target_position[1]
        dist_x = target_x - self.current_pose[0]
        dist_y = target_y - self.current_pose[1]
        distance_left = np.sqrt(dist_x ** 2 + dist_y ** 2)

        is_last = self.target_index >= len(self.waypoints) - 1
        precision = self.target_precision

        # If the current waypoint has been reached, move to the next one.
        if distance_left < precision:
            if is_last:
                # Last waypoint reached: keep target altitude and hold position.
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

        # Compute the angular difference between drone yaw and target direction.
        target_angle = np.arctan2(dist_y, dist_x)
        angle_left = target_angle - self.current_pose[5]
        angle_left = (angle_left + np.pi) % (2 * np.pi) - np.pi

        yaw_disturbance = clamp(
            angle_left,
            -self.MAX_YAW_DISTURBANCE,
            self.MAX_YAW_DISTURBANCE
        )

        # Move forward only when the drone is approximately aligned with the
        # target direction. Otherwise, rotate first.
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
        """
        Stop all propellers.
        """
        for motor in (
            self.front_left_motor,
            self.front_right_motor,
            self.rear_left_motor,
            self.rear_right_motor
        ):
            motor.setVelocity(0)

    def run(self):
        """
        Main drone control loop.
        """
        self.target_position = [0.0, 0.0, 0.0]
        self.target_altitude = HOME_ALT_MIN

        yaw_disturbance = 0.0
        pitch_disturbance = 0.0

        # The mission becomes armed once the first valid waypoint list is received.
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
                # If the mission has been aborted, ignore any new messages from
                # the receiver queue to avoid replacing the emergency route.
                while self.receiver is not None and self.receiver.getQueueLength() > 0:
                    self.receiver.nextPacket()
            else:
                self.consume_messages()

            if had_no_waypoints and self.waypoints:
                mission_armed = True
                print(f"[drone] Mission armed with {len(self.waypoints)} waypoints.")

            # =========================
            # B. CURRENT DRONE STATE
            # =========================

            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()

            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            # Store the initial home/base position once GPS is valid.
            if self.home_position is None and not np.isnan(x_pos):
                self.home_position = [x_pos, y_pos, HOME_ALT_MIN]

            # =========================
            # C. SAFETY MONITORING
            # =========================

            # During an active mission, monitor the vehicle in two ways:
            #   - distance: if the car falls more than MAX_CAR_DISTANCE m
            #     behind, the mission is aborted (vehicle lost / car stopped).
            #   - overtake: if the projection of the car position along the
            #     drone forward direction exceeds OVERTAKE_AHEAD_THRESHOLD for
            #     OVERTAKE_DEBOUNCE seconds, the mission is aborted (the car
            #     is not following the drone anymore).
            abort_reason = None

            if self.waypoints and self.car_node and mission_armed and not mission_aborted:
                car_position = self.car_node.getPosition()
                car_distance = np.sqrt(
                    (car_position[0] - x_pos) ** 2 +
                    (car_position[1] - y_pos) ** 2
                )

                if car_distance > MAX_CAR_DISTANCE:
                    print(f"[drone] Alert: vehicle lost at {car_distance:.1f} m. Aborting mission.")
                    abort_reason = "car_lost"

                else:
                    projection = self.car_projection_ahead(x_pos, y_pos)
                    now_t = self.getTime()

                    # Arm overtake detection only after the drone has caught
                    # up with the car (distance below OVERTAKE_ARM_DISTANCE).
                    # This avoids false positives at start-up, where the car
                    # waits at access_south while the drone is still at home
                    # behind it.
                    if (
                        not self.overtake_detection_armed
                        and car_distance < OVERTAKE_ARM_DISTANCE
                    ):
                        self.overtake_detection_armed = True
                        print(
                            f"[drone] Overtake detection armed "
                            f"(distance to car {car_distance:.2f} m)."
                        )

                    if (
                        self.overtake_detection_armed
                        and projection is not None
                        and projection > OVERTAKE_AHEAD_THRESHOLD
                    ):
                        if self.overtake_start_time is None:
                            self.overtake_start_time = now_t
                        elif now_t - self.overtake_start_time >= OVERTAKE_DEBOUNCE:
                            print(
                                f"[drone] Alert: vehicle overtook the drone "
                                f"(projection={projection:.1f} m). Aborting mission."
                            )
                            abort_reason = "car_overtake"
                    else:
                        self.overtake_start_time = None

            if abort_reason is not None:
                mission_aborted = True

                self.send_event({
                    "type": "mission_aborted",
                    "reason": abort_reason,
                    "time": self.getTime(),
                })

                # Emergency waypoint 1: return to base while preserving
                # the current flight altitude.
                return_wp = [
                    self.home_position[0],
                    self.home_position[1],
                    self.target_altitude
                ]

                # Emergency waypoint 2: descend once the drone is above base.
                landing_wp = [
                    self.home_position[0],
                    self.home_position[1],
                    HOME_ALT_MIN
                ]

                # Replace the original mission with the emergency route.
                self.waypoints = [return_wp, landing_wp]
                self.target_index = 0

                self.target_position[0] = return_wp[0]
                self.target_position[1] = return_wp[1]
                self.target_altitude = return_wp[2]

            # =========================
            # C.5. OBSTACLE SCENARIO
            # =========================

            # Collision-prevention scenario based on the camera's Recognition
            # output. While the obstacle is in view the drone hovers in place
            # and asks the obstacle car to clear the route. The scenario stays
            # inactive in worlds that do not contain a recognizable obstacle.
            obstacle_hold = False
            if mission_armed and not mission_aborted:
                seen_obstacle = self.obstacle_in_view()

                if self.obstacle_state == "armed" and seen_obstacle:
                    print("[drone] Obstacle detected by recognition. Hovering.")
                    self.trigger_obstacle_car("obstacle_seen")
                    self.send_event({
                        "type": "obstacle_detected",
                        "drone_position": [x_pos, y_pos, altitude],
                        "time": self.getTime(),
                    })
                    self.obstacle_state = "visible"

                elif self.obstacle_state == "visible" and not seen_obstacle:
                    print("[drone] Obstacle cleared from view. Resuming mission.")
                    self.send_event({
                        "type": "obstacle_cleared",
                        "time": self.getTime(),
                    })
                    self.obstacle_state = "cleared"

                obstacle_hold = (self.obstacle_state == "visible")

            # =========================
            # D. IDLE / MISSION CONTROL
            # =========================

            # Before any mission is received, keep the drone on the ground.
            if not mission_armed:
                self.stop_motors()
                continue

            if self.waypoints:
                if obstacle_hold:
                    # Force the drone to hover in place while the obstacle is
                    # still in the camera frame.
                    yaw_disturbance = 0.0
                    pitch_disturbance = 0.0
                # Move horizontally only after reaching a safe altitude close
                # to the target altitude.
                elif altitude > self.target_altitude - 0.3:
                    if self.getTime() - t1 > 0.1:
                        yaw_disturbance, pitch_disturbance = self.move_to_target()
                        t1 = self.getTime()
                else:
                    yaw_disturbance = 0.0
                    pitch_disturbance = 0.0
            else:
                # If no waypoints remain after arming, hover at the current position.
                self.target_position[0] = x_pos
                self.target_position[1] = y_pos
                yaw_disturbance = 0.0
                pitch_disturbance = 0.0

            # =========================
            # E. PERIODIC DEBUG LOGGING
            # =========================

            now = self.getTime()

            if now - last_log > 1.0:
                if self.waypoints:
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
                            f"pitchD={pitch_disturbance:.2f}"
                        )
                else:
                    print(f"[drone] Hovering at ({x_pos:.1f},{y_pos:.1f},{altitude:.1f})")

                last_log = now

            # =========================
            # F. LOW-LEVEL MOTOR CONTROL
            # =========================

            # Stabilization inputs derived from current orientation.
            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration
            pitch_input = (
                self.K_PITCH_P * clamp(pitch, -1, 1) +
                pitch_acceleration +
                pitch_disturbance
            )
            yaw_input = yaw_disturbance

            # Altitude control input.
            clamped_diff_alt = clamp(
                self.target_altitude - altitude + self.K_VERTICAL_OFFSET,
                -1,
                1
            )
            vertical_input = self.K_VERTICAL_P * pow(clamped_diff_alt, 3.0)

            # Motor mixing for the quadrotor configuration.
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
