"""
CoppeliaSim – Scene Generator + Colour Detection + Arm Sorting
--------------------------------------------------------------
Requirements:
    pip install coppeliasim-zmqremoteapi-client opencv-python numpy

How to use:
    1. Open CoppeliaSim
    2. Hit Play ▶ in CoppeliaSim
    3. Run this script
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import numpy as np
import time
import os
import glob

# ──────────────────────────────────────────────────────────────
# CONNECT
# ──────────────────────────────────────────────────────────────

client = RemoteAPIClient()
sim    = client.getObject('sim')

# ──────────────────────────────────────────────────────────────
# SCENE PARAMETERS
# ──────────────────────────────────────────────────────────────

CUBE_SIZE = 0.055

CUBE_SPAWN = {
    "red":    [ 0.35,  0.15, CUBE_SIZE / 2],
    "green":  [ 0.35,  0.05, CUBE_SIZE / 2],
    "blue":   [ 0.35, -0.05, CUBE_SIZE / 2],
    "yellow": [ 0.35, -0.15, CUBE_SIZE / 2],
}

CUBE_COLOURS = {
    "red":    [1.0, 0.05, 0.05],
    "green":  [0.05, 0.9, 0.05],
    "blue":   [0.05, 0.05, 1.0],
    "yellow": [1.0, 0.95, 0.0],
}

DROP_POSITIONS = {
    "red":    [-0.35,  0.20, CUBE_SIZE / 2],
    "green":  [-0.35,  0.07, CUBE_SIZE / 2],
    "blue":   [-0.35, -0.07, CUBE_SIZE / 2],
    "yellow": [-0.35, -0.20, CUBE_SIZE / 2],
}

CAM_POS    = [0.35, 0.0, 0.7]
CAM_ORIENT = [np.pi, 0.0, 0.0]
CAM_RES    = [512, 512]
CAM_FOV    = 80.0

ARM_MODEL_NAME = "7 DoF manipulator.ttm"
SEARCH_ROOTS = [
    r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu",
]

GRIPPER_OPEN  =  0.04
GRIPPER_CLOSE =  0.0

HOME_ANGLES = [0.0, np.radians(-30), 0.0, np.radians(-90), 0.0, np.radians(60), 0.0]

COLOUR_RANGES = {
    "red": [
        (np.array([0,   140,  80]), np.array([10,  255, 255])),
        (np.array([168, 140,  80]), np.array([179, 255, 255])),
    ],
    "green":  [(np.array([45,  80,  60]), np.array([85,  255, 255]))],
    "blue":   [(np.array([100, 120,  60]), np.array([130, 255, 255]))],
    "yellow": [(np.array([22,  130, 100]), np.array([38,  255, 255]))],
}
MIN_BLOB_AREA = 300

# ──────────────────────────────────────────────────────────────
# FIND ARM MODEL
# ──────────────────────────────────────────────────────────────

def find_arm_model() -> str:
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        matches = glob.glob(os.path.join(root, "**", ARM_MODEL_NAME), recursive=True)
        if matches:
            print(f"[SCENE] Found arm model: {matches[0]}")
            return matches[0]
    raise FileNotFoundError(
        f"Could not find '{ARM_MODEL_NAME}'.\n"
        f"Check SEARCH_ROOTS at the top of this script."
    )

# ──────────────────────────────────────────────────────────────
# SCENE BUILDING
# ──────────────────────────────────────────────────────────────
def create_cube(name, position, colour_rgb):
    handle = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid,
        [CUBE_SIZE, CUBE_SIZE, CUBE_SIZE],
        0
    )
    sim.setObjectAlias(handle, name)
    sim.setObjectPosition(handle, -1, position)
    sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, colour_rgb)
    # removed visibility layer — let sensor see everything
    sim.setObjectInt32Param(handle, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, 1)
    return handle

def create_drop_zone_marker(name, position, colour_rgb):
    handle = sim.createPrimitiveShape(
        sim.primitiveshape_cylinder,
        [0.08, 0.08, 0.003],
        0
    )
    sim.setObjectAlias(handle, name)
    sim.setObjectPosition(handle, -1, position)
    sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, colour_rgb)
    sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, 0)
    return handle


def create_vision_sensor():
    int_params   = [CAM_RES[0], CAM_RES[1], 0, 0, 0, 0, 0, 0]
    float_params = [
        np.radians(CAM_FOV), 0.01, 5.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    handle = sim.createVisionSensor(1, int_params, float_params)  # ← 1 enables explicit handling
    sim.setObjectAlias(handle, "ColourCamera")
    sim.setObjectPosition(handle, -1, CAM_POS)
    sim.setObjectOrientation(handle, -1, CAM_ORIENT)
    return handle


def load_arm():
    model_path = find_arm_model()
    arm_root   = sim.loadModel(model_path)
    sim.setObjectPosition(arm_root, -1, [0.0, 0.0, 0.0])

    joints = []
    stack  = [arm_root]
    while stack:
        obj = stack.pop()
        if sim.getObjectType(obj) == sim.object_joint_type:
            joints.append(obj)
        try:
            i = 0
            while True:
                child = sim.getObjectChild(obj, i)
                if child == -1:
                    break
                stack.append(child)
                i += 1
        except Exception:
            pass

    print(f"[SCENE] Found {len(joints)} joints in arm model.")

    gripper = -1
    for j in joints:
        alias = sim.getObjectAlias(j, -1).lower()
        if any(k in alias for k in ["gripper", "finger", "tool"]):
            gripper = j
            break
    if gripper == -1 and joints:
        gripper = joints[-1]

    target = -1
    for tname in ["/Target", "/IK_target"]:
        try:
            target = sim.getObject(tname, {"noError": True})
            if target != -1:
                break
        except Exception:
            pass
    if target == -1:
        target = sim.createDummy(0.02)
        sim.setObjectAlias(target, "IK_target")
        sim.setObjectPosition(target, -1, [0.35, 0.15, 0.3])
        print("[SCENE] Created IK_target dummy.")

    arm_joints = [j for j in joints if j != gripper]
    return arm_root, arm_joints, gripper, target


def build_scene():
    print("[SCENE] Building scene …")
    handles = {}

    handles["cubes"] = {}
    for colour, pos in CUBE_SPAWN.items():
        h = create_cube(f"Cube_{colour}", pos, CUBE_COLOURS[colour])
        handles["cubes"][colour] = h
        print(f"[SCENE]   Cube '{colour}' at {pos}")

    handles["zones"] = {}
    for colour, pos in DROP_POSITIONS.items():
        faded = [min(1.0, c * 0.5 + 0.3) for c in CUBE_COLOURS[colour]]
        h = create_drop_zone_marker(f"Zone_{colour}", pos, faded)
        handles["zones"][colour] = h
        print(f"[SCENE]   Zone '{colour}' at {pos}")

    handles["vision"] = create_vision_sensor()
    print("[SCENE]   Vision sensor created.")

    arm_root, arm_joints, gripper, target = load_arm()
    handles["arm_root"] = arm_root
    handles["joints"]   = arm_joints
    handles["gripper"]  = gripper
    handles["target"]   = target
    print(f"[SCENE]   Arm loaded — joints: {len(arm_joints)}, gripper: {gripper}, target: {target}")

    print("[SCENE] Scene ready.\n")
    return handles

# ──────────────────────────────────────────────────────────────
# VISION / COLOUR DETECTION
# ──────────────────────────────────────────────────────────────

def get_camera_frame(vision_handle):
    try:
        sim.handleVisionSensor(vision_handle)
    except Exception as e:
        print(f"[CAM] handleVisionSensor error: {e}")

    try:
        img_data, resolution = sim.getVisionSensorImg(vision_handle)
        if not img_data:
            print("[CAM] No image data returned.")
            return None

        img = np.frombuffer(img_data, dtype=np.uint8)
        img = img.reshape((resolution[1], resolution[0], 3))
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # DEBUG – save one frame to disk so we can inspect it
        cv2.imwrite("debug_frame.png", img)

        return img
    except Exception as e:
        print(f"[CAM] getVisionSensorImg error: {e}")
        return None
     
def detect_colour(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    best_colour, best_area, best_centroid = "unknown", 0, None

    for colour, ranges in COLOUR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area > MIN_BLOB_AREA and area > best_area:
            best_area   = area
            best_colour = colour
            M = cv2.moments(largest)
            if M["m00"] != 0:
                best_centroid = (int(M["m10"] / M["m00"]),
                                 int(M["m01"] / M["m00"]))

    return best_colour, best_centroid


def annotate_frame(frame, colour, centroid):
    COLOUR_BGR = {
        "red":    (0,   0,   255),
        "green":  (0,   200,   0),
        "blue":   (255,  80,   0),
        "yellow": (0,   220, 220),
        "unknown":(180, 180, 180),
    }
    bgr = COLOUR_BGR.get(colour, (180, 180, 180))
    if centroid:
        cv2.circle(frame, centroid, 10, bgr, -1)
        cv2.circle(frame, centroid, 12, (255, 255, 255), 2)
    cv2.putText(frame, f"Detected: {colour.upper()}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr, 2)
    return frame

# ──────────────────────────────────────────────────────────────
# ARM CONTROL
# ──────────────────────────────────────────────────────────────

def set_joints(handles, angles, pause=1.5):
    for i, j in enumerate(handles["joints"]):
        if i < len(angles):
            sim.setJointTargetPosition(j, angles[i])
    time.sleep(pause)


def move_to_home(handles):
    print("[ARM] → Home")
    set_joints(handles, HOME_ANGLES)


def open_gripper(handles):
    if handles["gripper"] != -1:
        sim.setJointTargetPosition(handles["gripper"], GRIPPER_OPEN)
    time.sleep(0.4)


def close_gripper(handles):
    if handles["gripper"] != -1:
        sim.setJointTargetPosition(handles["gripper"], GRIPPER_CLOSE)
    time.sleep(0.4)


def move_target_to(handles, position, pause=1.2):
    sim.setObjectPosition(handles["target"], -1, position)
    time.sleep(pause)


def pick(handles, pos):
    approach = [pos[0], pos[1], pos[2] + 0.18]
    contact  = [pos[0], pos[1], pos[2] + 0.005]
    open_gripper(handles)
    move_target_to(handles, approach)
    move_target_to(handles, contact)
    close_gripper(handles)
    move_target_to(handles, approach)


def place(handles, pos):
    approach = [pos[0], pos[1], pos[2] + 0.18]
    release  = [pos[0], pos[1], pos[2] + 0.01]
    move_target_to(handles, approach)
    move_target_to(handles, release)
    open_gripper(handles)
    move_target_to(handles, approach)

# ──────────────────────────────────────────────────────────────
# SORTING LOGIC
# ──────────────────────────────────────────────────────────────

def sort_cube(handles, colour):
    if colour == "unknown" or colour not in handles["cubes"]:
        return
    pick_pos = list(sim.getObjectPosition(handles["cubes"][colour], -1))
    drop_pos = DROP_POSITIONS[colour]
    print(f"[SORT] Picking '{colour}' …")
    pick(handles, pick_pos)
    print(f"[SORT] Placing '{colour}' …")
    place(handles, drop_pos)
    move_to_home(handles)
    print(f"[SORT] '{colour}' done.\n")

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    state = sim.getSimulationState()
    if state == sim.simulation_stopped:
        print("[INIT] Starting simulation …")
        sim.startSimulation()
        time.sleep(1.5)

    handles = build_scene()
    time.sleep(0.5)
    move_to_home(handles)

    sorted_colours  = set()
    detection_pause = 2.5
    last_sort_time  = 0.0

    print("[MAIN] Running. Press 'q' in the camera window to quit.\n")

    while True:
        frame = get_camera_frame(handles["vision"])

        if frame is None:
            # Show a black placeholder so the window stays open
            placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera...", (60, 256),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.imshow("Colour Camera", placeholder)
        else:
            colour, centroid = detect_colour(frame)
            cv2.imshow("Colour Camera", annotate_frame(frame.copy(), colour, centroid))

            now = time.time()
            if (colour != "unknown"
                    and colour not in sorted_colours
                    and (now - last_sort_time) > detection_pause):
                sort_cube(handles, colour)
                sorted_colours.add(colour)
                last_sort_time = time.time()

            if len(sorted_colours) == len(CUBE_SPAWN):
                print("[MAIN] All cubes sorted!")
                time.sleep(2)
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("[MAIN] Done.")


if __name__ == "__main__":
    main()
