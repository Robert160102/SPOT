"""
Follower vehicle controller for the SPOT Webots simulation.

This controller receives navigation targets from the drone/supervisor through
a Webots Receiver node and drives the vehicle towards those targets.

The vehicle model is intentionally simplified: the goal is not to implement
a fully optimized autonomous driving system, but to validate the drone-guided
navigation flow inside the simulation.
"""

from vehicle import Driver
import math
import time


# =========================
# CONTROLLER INITIALIZATION
# =========================

driver = Driver()
TIME_STEP = int(driver.getBasicTimeStep())

# Development flag used to enable periodic debug information.
dev = True


# =========================
# CONTROL PARAMETERS
# =========================

# The supervisor/drone sends dense target points with lookahead.
# Therefore, the vehicle only enters the braking zone near the final spot.
DISTANCIA_FRENADO = 1.0
DISTANCIA_PARADA = 0.3

# Default vehicle cruising speed in m/s.
VELOCIDAD_CRUCERO_DEFAULT = 5.0

# Active cruise speed. The supervisor may override this value at runtime by
# appending a third field to the target message (e.g., during the overtake
# failure scenario, where the car is expected to drive faster).
velocidad_crucero = VELOCIDAD_CRUCERO_DEFAULT

# Steering controller gain and physical steering limit.
KP_STEER = 1.2
MAX_STEERING_ANGLE = math.radians(30)

# Communication timeout. If no target is received for this time,
# the vehicle stops for safety.
TIMEOUT_SENAL = 2.0
DEBUG_LOG_INTERVAL = 0.5

# Reverse maneuver parameters.
# This is used when the vehicle faces almost the opposite direction
# of the target and needs to recover from a sharp turn.
REVERSE_DURATION = 2.0
REVERSE_SPEED = -2.0
REVERSE_ERROR_THRESHOLD = math.radians(150)
REVERSE_DISTANCE_THRESHOLD = 10.0


# =========================
# DEVICE INITIALIZATION
# =========================

# GPS provides the current vehicle position inside the Webots world.
gps = driver.getDevice("gps")
gps.enable(TIME_STEP)

# Compass provides orientation data used to compute the heading angle.
compass = driver.getDevice("compass")
compass.enable(TIME_STEP)

# Receiver node used to obtain target coordinates from the drone/supervisor.
receiver = driver.getDevice("receiver")
receiver.enable(TIME_STEP)

# Channel 1 must match the emitter channel used by the drone/supervisor.
receiver.setChannel(1)

print("Follower vehicle controller active. Waiting for navigation targets.")


# =========================
# STATE VARIABLES
# =========================

target_x = None
target_y = None

last_message_time = time.time()
last_debug_log = 0.0

previous_target_x = None
previous_target_y = None

# Timestamp until which the reverse maneuver remains active.
reverse_until = 0.0


# =========================
# UTILITY FUNCTIONS
# =========================

def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].

    This avoids discontinuities when comparing orientations close to
    the -pi / pi boundary.
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def heading_from_compass(compass_values):
    """
    Compute the vehicle heading angle from Webots compass values.
    """
    return math.atan2(compass_values[1], compass_values[0])


# =========================
# MAIN CONTROL LOOP
# =========================

while driver.step() != -1:

    # =========================
    # A. TARGET RECEPTION
    # =========================

    # Process all pending messages from the receiver queue.
    # Each message is expected to contain target coordinates in the format
    # "x,y" or, when the supervisor wants to override the cruise speed,
    # "x,y,speed".
    while receiver.getQueueLength() > 0:
        message = receiver.getString()
        receiver.nextPacket()

        try:
            parts = message.split(',')
            new_target_x = float(parts[0])
            new_target_y = float(parts[1])

            # Update target only if the received position is different.
            if new_target_x != previous_target_x or new_target_y != previous_target_y:
                target_x = new_target_x
                target_y = new_target_y
                previous_target_x = target_x
                previous_target_y = target_y

                if dev:
                    print(f"[car] New target received: ({target_x:.2f}, {target_y:.2f})")

            # Optional cruise speed override. When absent, fall back to the
            # default cruise speed.
            if len(parts) >= 3:
                try:
                    new_speed = float(parts[2])
                    if new_speed != velocidad_crucero:
                        velocidad_crucero = new_speed
                        if dev:
                            print(f"[car] Cruise speed overridden to {velocidad_crucero:.2f} m/s")
                except ValueError:
                    pass
            else:
                if velocidad_crucero != VELOCIDAD_CRUCERO_DEFAULT:
                    velocidad_crucero = VELOCIDAD_CRUCERO_DEFAULT
                    if dev:
                        print(f"[car] Cruise speed reset to default {velocidad_crucero:.2f} m/s")

            last_message_time = time.time()

        except Exception:
            print(f"[car] Invalid receiver message: {message}")

    # =========================
    # B. SAFETY CHECKS
    # =========================

    # If no target has been received yet, keep the vehicle stopped.
    if target_x is None or target_y is None:
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(1.0)
        continue

    # If communication is lost, stop the vehicle and clear the target.
    if (time.time() - last_message_time) > TIMEOUT_SENAL:
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(1.0)
        target_x = target_y = None
        continue

    # =========================
    # C. CURRENT VEHICLE STATE
    # =========================

    # Read current vehicle position.
    pos = gps.getValues()
    car_x, car_y = pos[0], pos[1]

    # Compute vector and distance to the current target.
    dx = target_x - car_x
    dy = target_y - car_y
    distance = math.hypot(dx, dy)

    # Compute current heading and heading error.
    compass_values = compass.getValues()
    car_heading = normalize_angle(heading_from_compass(compass_values))
    target_angle = math.atan2(dx, dy)
    heading_error = normalize_angle(target_angle - car_heading)

    # =========================
    # D. REVERSE MANEUVER
    # =========================

    # Trigger reverse maneuver if the vehicle is close to the target
    # but facing almost the opposite direction.
    if (
        time.time() > reverse_until
        and abs(heading_error) > REVERSE_ERROR_THRESHOLD
        and distance < REVERSE_DISTANCE_THRESHOLD
    ):
        reverse_until = time.time() + REVERSE_DURATION

        if dev:
            print("[car] Starting reverse recovery maneuver.")

    # Execute reverse maneuver while the timer is active.
    if time.time() < reverse_until:
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(REVERSE_SPEED)

        # Turn the steering wheel in the opposite direction of the error
        # to recover vehicle alignment.
        reverse_steer = -math.copysign(MAX_STEERING_ANGLE, heading_error)
        driver.setSteeringAngle(reverse_steer)

        if dev and (time.time() - last_debug_log) >= DEBUG_LOG_INTERVAL:
            last_debug_log = time.time()
            print(
                f"[car] Reverse maneuver "
                f"pos=({car_x:.2f},{car_y:.2f}) "
                f"error={heading_error:.2f} "
                f"steer={reverse_steer:.2f} "
                f"speed={REVERSE_SPEED:.2f}"
            )

        # Skip normal control while reversing.
        continue

    # =========================
    # E. NORMAL STEERING CONTROL
    # =========================

    # Smooth steering command based on the heading error.
    # tanh limits aggressive steering changes.
    steer_cmd = KP_STEER * math.tanh(heading_error)

    # Clamp steering command to vehicle physical limits.
    steering_angle = max(
        -MAX_STEERING_ANGLE,
        min(MAX_STEERING_ANGLE, steer_cmd)
    )

    driver.setSteeringAngle(steering_angle)

    # =========================
    # F. SPEED CONTROL
    # =========================

    # Reduce speed when the vehicle is poorly aligned with the target.
    # alignment = 1 means perfect alignment, 0 means perpendicular/opposite.
    alignment = max(0.0, 1.0 - abs(heading_error) / math.pi)
    speed_cmd = velocidad_crucero * alignment

    # If the heading error is very large and the target is close,
    # advance slowly to avoid overshooting the target.
    if abs(heading_error) > math.radians(150) and distance < 5.0:
        speed_cmd = 1.0 / 3.6
    else:
        # Maintain a minimum speed so the vehicle can complete wide turns.
        speed_cmd = max(speed_cmd, 1.5)

    # =========================
    # G. BRAKING CONTROL
    # =========================

    if distance < DISTANCIA_PARADA:
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(1.0)
        speed_cmd = 0.0
        brake_cmd = 1.0

    elif distance < DISTANCIA_FRENADO:
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(0.2)
        speed_cmd = 0.0
        brake_cmd = 0.2

    else:
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(speed_cmd)
        brake_cmd = 0.0

    # =========================
    # H. DEBUG LOGGING
    # =========================

    if dev and (time.time() - last_debug_log) >= DEBUG_LOG_INTERVAL:
        last_debug_log = time.time()
        print(
            f"[car] pos=({car_x:.2f}, {car_y:.2f}) "
            f"target=({target_x:.2f}, {target_y:.2f}) "
            f"distance={distance:.2f} "
            f"heading={car_heading:.2f} "
            f"target_angle={target_angle:.2f} "
            f"error={heading_error:.2f} "
            f"steer={steering_angle:.2f} "
            f"speed={speed_cmd:.2f} "
            f"brake={brake_cmd:.2f}"
        )