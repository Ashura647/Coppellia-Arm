from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
client = RemoteAPIClient()
sim = client.getObject('sim')
import keyboard

armjoint1,armjoint2,armjoint3,armjoint4,armjoint5,armjoint6=( 
    sim.getObject('/PArm/joint1'),
    sim.getObject('/PArm/joint1/joint2'),
    sim.getObject('/PArm/joint1/joint2/joint3'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4/joint5'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4/joint5/joint6'),
)

fingerjoint1  = sim.getObject('/PGripStraight/motor')
conveyor      = sim.getObject('/efficientConveyor')
sensor        = sim.getObject('/proximitySensor')
visionSensor  = sim.getObject('/sphericalVisionRGBAndDepth/sensorRGB')
gripperTip    = sim.getObject('/PGripStraight/connector')

# ── Size thresholds (metres) — 0.08 is your normal cube size ─────────────────
NORMAL_SIZE_MIN = 0.03
NORMAL_SIZE_MAX = 0.09

# ── Colour thresholds — update these after first calibration run ──────────────
COLOUR_THRESHOLDS = {
    'red':   {'r': (0.5, 1.0), 'g': (0.0, 0.4), 'b': (0.0, 0.4)},
    'green': {'r': (0.0, 0.4), 'g': (0.5, 1.0), 'b': (0.0, 0.4)},
    'blue':  {'r': (0.0, 0.4), 'g': (0.0, 0.4), 'b': (0.5, 1.0)},
}

# ── Get object size via bounding box ─────────────────────────────────────────
def get_object_size(handle):
    try:
        x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_x) \
          - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_x)
        y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_y) \
          - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_y)
        z = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z) \
          - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z)
        return max(x, y, z)
    except Exception as e:
        print(f"  [SIZE] Error: {e}")
        return None

# ── Read colour from vision sensor ────────────────────────────────────────────
# def get_colour():
#     sim.handleVisionSensor(visionSensor)
#     result = sim.readVisionSensor(visionSensor)
#     print(f"  [CAL] Raw sensor result: {result}")

#     if result == -1 or result is None:
#         print("  [CAL] Sensor returned -1 — not detecting anything")
#         return 'unknown'

#     if not isinstance(result, (list, tuple)) or len(result) < 2:
#         print(f"  [CAL] Unexpected result format: {result}")
#         return 'unknown'

#     packet = result[1]
#     if not packet or len(packet) < 14:
#         print("  [CAL] Packet too short or empty")
#         return 'unknown'

#     r = packet[11]
#     g = packet[12]
#     b = packet[13]
#     print(f"  [CAL] R={r:.3f}, G={g:.3f}, B={b:.3f}")

#     for colour, ranges in COLOUR_THRESHOLDS.items():
#         if (ranges['r'][0] <= r <= ranges['r'][1] and
#                 ranges['g'][0] <= g <= ranges['g'][1] and
#                 ranges['b'][0] <= b <= ranges['b'][1]):
#             return colour
#     return 'unknown'

def get_colour(handle):
    try:
        result, colour = sim.getShapeColor(handle, None, sim.colorcomponent_ambient_diffuse)
        r, g, b = colour[0], colour[1], colour[2]
        print(f"  [CAL] Object colour — R={r:.3f}, G={g:.3f}, B={b:.3f}")
        
        if r > 0.5 and g < 0.4 and b < 0.4:
            return 'red'
        elif g > 0.5 and r < 0.4 and b < 0.4:
            return 'green'
        elif b > 0.5 and r < 0.4 and g < 0.4:
            return 'blue'
        else:
            return 'unknown'
    except Exception as e:
        print(f"  [CAL] Colour error: {e}")
        return 'unknown'   
    
# ── Move arm to scan position (cube held under vision sensor) ─────────────────
def scan_position():
    sim.setJointTargetPosition(armjoint1, 0)
    sim.setJointTargetPosition(armjoint2, 0.8)
    for _ in range(300):
        sim.step()

# ── Classify object by size and colour ───────────────────────────────────────
# def classify(handle):
#     size = get_object_size(handle)
#     print(f"  [CAL] Size = {size}")

#     size_ok = (size is not None and NORMAL_SIZE_MIN <= size <= NORMAL_SIZE_MAX)
#     if not size_ok:
#         print(f"  ⚠  DEFECT — size {size} outside normal range [{NORMAL_SIZE_MIN},{NORMAL_SIZE_MAX}]")
#         return 'defect'

#     colour = get_colour()
#     if colour not in ('red', 'green', 'blue'):
#         print(f"  ⚠  DEFECT — unrecognised colour")
#         return 'defect'

#     print(f"  ✓  Normal — colour={colour}, size={size:.4f}m")
#     return colour

def classify(handle):
    size = get_object_size(handle)
    print(f"  [CAL] Size = {size}")
    size_ok = (size is not None and NORMAL_SIZE_MIN <= size <= NORMAL_SIZE_MAX)
    if not size_ok:
        print(f"  ⚠  DEFECT — bad size")
        return 'defect'
    colour = get_colour(handle)  # ← pass handle directly
    if colour not in ('red', 'green', 'blue'):
        print(f"  ⚠  DEFECT — unrecognised colour")
        return 'defect'
    print(f"  ✓  Normal — {colour}, size={size:.4f}m")
    return colour

def startingpostion():
    sim.setJointTargetPosition(armjoint1, 0)
    sim.setJointTargetPosition(armjoint2, 0)
    sim.setJointTargetPosition(armjoint3, 0)
    sim.setJointTargetPosition(armjoint4, 0)
    sim.setJointTargetPosition(armjoint5, 0)
    sim.setJointTargetPosition(armjoint6, 0)
    for _ in range(400):
        sim.step()

def detection(n):
    sim.setJointTargetPosition(armjoint2, 1.4)
    sim.setJointTargetPosition(armjoint3, 1.77)
    sim.setJointTargetPosition(armjoint5, -1.4)
    for _ in range(200):
        sim.step()
    sim.setJointTargetPosition(fingerjoint1, -0.044)
    for _ in range(80):
        sim.step()
    sim.setObjectParent(n, gripperTip, True)

def red(n):
    sim.setJointTargetPosition(armjoint1, 1)
    for _ in range(200):
        sim.step()
    sim.setJointTargetPosition(armjoint5, -1.5)
    for _ in range(500):
        sim.step()
    sim.setJointTargetPosition(armjoint2, 1)
    for _ in range(200):
        sim.step()
    sim.setJointTargetPosition(fingerjoint1, 0)
    sim.setObjectParent(n, -1, True)
    for _ in range(200):
        sim.step()

def blue(n):
    sim.setJointTargetPosition(armjoint1, 1.6)
    for _ in range(200):
        sim.step()
    sim.setJointTargetPosition(armjoint5, -1.5)
    for _ in range(500):
        sim.step()
    sim.setJointTargetPosition(armjoint2, 1)
    for _ in range(200):
        sim.step()
    sim.setJointTargetPosition(fingerjoint1, 0)
    sim.setObjectParent(n, -1, True)
    for _ in range(200):
        sim.step()

def green(n):
    sim.setJointTargetPosition(armjoint1, 2.2)
    for _ in range(200):
        sim.step()
    sim.setJointTargetPosition(armjoint5, -1.5)
    for _ in range(500):
        sim.step()
    sim.setJointTargetPosition(armjoint2, 1)
    for _ in range(200):
        sim.step()
    sim.setJointTargetPosition(fingerjoint1, 0)
    sim.setObjectParent(n, -1, True)
    for _ in range(200):
        sim.step()

def defect(n):
    sim.setJointTargetPosition(armjoint1, -1.0)
    for _ in range(200):
        sim.step()
    sim.setJointTargetPosition(armjoint5, -1.5)
    for _ in range(500):
        sim.step()
    sim.setJointTargetPosition(armjoint2, 1)
    for _ in range(200):
        sim.step()
    sim.setJointTargetPosition(fingerjoint1, 0)
    sim.setObjectParent(n, -1, True)
    for _ in range(200):
        sim.step()


sim.setStepping(True)
sim.startSimulation()

print("To end, press Q")

sim.setObjectFloatParam(armjoint1, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint1, sim.jointfloatparam_maxaccel, 0.1)
sim.setObjectFloatParam(armjoint5, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint5, sim.jointfloatparam_maxaccel, 0.1)
sim.setObjectFloatParam(armjoint3, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint3, sim.jointfloatparam_maxaccel, 0.1)
sim.setObjectFloatParam(armjoint2, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint2, sim.jointfloatparam_maxaccel, 0.1)

sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.017}))

grasped = False
while True:

    if keyboard.is_pressed('q'):
        print("Stopping simulation...")
        sim.stopSimulation()
        break

    detected, distance, detectedObjloc, detectedObjHandle, two22 = sim.readProximitySensor(sensor)

    if detected and not grasped:
        grasped = True
        print("─" * 40)
        print("Object detected! Stopping conveyor")
        sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.0}))

        # 1. Pick up
        detection(detectedObjHandle)
        for _ in range(400):
            sim.step()

        # 2. Move to scan position (under vision sensor)
        print("  Moving to scan position...")
        scan_position()

        # 3. Classify (size + colour)
        print("  Classifying object...")
        category = classify(detectedObjHandle)

        # 4. Return home
        startingpostion()
        for _ in range(400):
            sim.step()

        # 5. Sort
        print(f"  → Sorting to: {category.upper()} bin")
        if category == 'red':
            red(detectedObjHandle)
        elif category == 'blue':
            blue(detectedObjHandle)
        elif category == 'green':
            green(detectedObjHandle)
        else:
            defect(detectedObjHandle)

        for _ in range(400):
            sim.step()

        startingpostion()
        for _ in range(400):
            sim.step()

        sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.017}))
        grasped = False

    sim.step()

sim.stopSimulation()