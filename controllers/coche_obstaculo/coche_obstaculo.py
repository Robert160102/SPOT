"""
Obstacle vehicle controller for the collision-prevention scenario.

The "obstaculo" car drives across the main north-south road in two stages:

  1. At simulation start it drives forward until it reaches the midpoint of
     the crossing, where it stops and parks across the road.
  2. It waits there until the drone reports it has spotted the car. After
     receiving that signal it waits a fixed delay (representing the round
     trip with the server / operator decision) and then drives forward again
     until it has fully cleared the route.

The drone uses camera Recognition to spot this car. While it sees the car it
hovers; once the car has moved out of frame, the drone resumes its mission.

A boolean flag at the top of the file decides whether the scenario is active.
If the flag is False the car stays put forever, which allows the same world to
be reused for non-obstacle scenarios.
"""

from vehicle import Driver


# =========================
# SCENARIO FLAG
# =========================

# Set to False to disable the obstacle scenario in the current world. The car
# will remain parked and ignore any message coming from the drone.
OBSTACLE_ACTIVE = True


# =========================
# MOTION PARAMETERS
# =========================

# Cruising speed used while the car is moving, in m/s.
CRUISE_SPEED = 4.0

# World X coordinate of the midpoint where the car stops and waits for the
# drone signal. Matches the design point on the crossing (~0.99).
MID_X = 0.995779

# World X coordinate at which the car must finally stop after being told to
# clear the route.
END_X = -22.4128

# Seconds the car waits after receiving the drone signal before driving away.
# Represents the time the operator/server needs to acknowledge the alarm.
SIGNAL_DELAY = 10.0

# Communication channel shared with the drone's "emitter_obstaculo".
RECEIVER_CHANNEL = 4

# Event string the drone emits when Recognition first reports the car.
EVENT_SEEN = "obstacle_seen"


# =========================
# CONTROLLER INITIALIZATION
# =========================

driver = Driver()
TIME_STEP = int(driver.getBasicTimeStep())

gps = driver.getDevice("gps")
gps.enable(TIME_STEP)

receiver = driver.getDevice("receiver")
receiver.enable(TIME_STEP)
receiver.setChannel(RECEIVER_CHANNEL)


# =========================
# STATE
# =========================

# Possible states:
#   "moving_to_mid"   - driving from the start position towards the midpoint.
#   "waiting_signal"  - parked at the midpoint, waiting for the drone to
#                       confirm detection.
#   "waiting_delay"   - drone signal received, holding for SIGNAL_DELAY
#                       seconds before resuming motion.
#   "moving_to_end"   - clearing the route towards END_X.
#   "done"            - reached END_X, parked again.
state = "moving_to_mid"

# Timestamp when the drone signal was received. Used to time the delay before
# the car starts moving again.
signal_time = None

print(
    f"[obstaculo] Controller active. OBSTACLE_ACTIVE={OBSTACLE_ACTIVE}, "
    f"SIGNAL_DELAY={SIGNAL_DELAY}s."
)

driver.setSteeringAngle(0.0)
driver.setCruisingSpeed(0.0)
driver.setBrakeIntensity(1.0)


# =========================
# MAIN LOOP
# =========================

while driver.step() != -1:

    sim_time = driver.getTime()

    # Drain the receiver queue. Even when the scenario is disabled we still
    # consume messages so they do not accumulate.
    while receiver.getQueueLength() > 0:
        msg = receiver.getString()
        receiver.nextPacket()

        if not OBSTACLE_ACTIVE:
            continue

        if msg == EVENT_SEEN and state == "waiting_signal":
            state = "waiting_delay"
            signal_time = sim_time
            print(
                f"[obstaculo] Drone reports detection. Waiting "
                f"{SIGNAL_DELAY:.1f}s before clearing route."
            )

    if not OBSTACLE_ACTIVE:
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(1.0)
        continue

    x = gps.getValues()[0]

    # =========================
    # STATE TRANSITIONS
    # =========================

    if state == "moving_to_mid" and x <= MID_X:
        state = "waiting_signal"
        print(f"[obstaculo] Reached midpoint X={x:.2f}. Waiting for drone signal.")

    elif state == "waiting_delay" and sim_time - signal_time >= SIGNAL_DELAY:
        state = "moving_to_end"
        print("[obstaculo] Delay elapsed. Clearing route.")

    elif state == "moving_to_end" and x <= END_X:
        state = "done"
        print(f"[obstaculo] Reached end X={x:.2f}. Parking.")

    # =========================
    # ACTUATION
    # =========================

    # The car is spawned rotated pi around Z, so its forward direction maps
    # to decreasing X in world coordinates. Positive cruise speed drives the
    # car towards -X.
    if state in ("moving_to_mid", "moving_to_end"):
        driver.setBrakeIntensity(0.0)
        driver.setSteeringAngle(0.0)
        driver.setCruisingSpeed(CRUISE_SPEED)
    else:
        # waiting_signal, waiting_delay, done -> stay still.
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(1.0)
