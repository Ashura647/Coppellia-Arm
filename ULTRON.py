from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import random
import numpy as np
import keyboard
import json
import os

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
#
#  USE_ML          — set True to enable the neural network classifier.
#                    When enabled, after MIN_SAMPLES_BEFORE_ML training examples
#                    have been collected the NN will run alongside the rule
#                    classifier. If NN confidence > ML_CONFIDENCE_THRESHOLD the
#                    NN result overrides the rule result; otherwise the rule
#                    result is used and the disagreement is logged so the NN can
#                    learn from it.
#
#  LOAD_MODEL      — set True to load previously saved weights + training data
#                    from WEIGHTS_FILE / TRAINING_FILE before starting.
#                    Lets the arm continue learning across sessions.
#
#  SAVE_EVERY_N    — save weights + training data to disk every N objects.
#                    Set to 1 to save after every single pick-and-place cycle.
#
#  WEIGHTS_FILE    — path to the .npz file that stores network weights.
#  TRAINING_FILE   — path to the .json file that stores labelled training data.
#
# ═══════════════════════════════════════════════════════════════════════════════
USE_ML                  = True
LOAD_MODEL              = True        # load saved weights / data on startup
SAVE_EVERY_N            = 5           # save to disk every N objects sorted
MIN_SAMPLES_BEFORE_ML   = 20          # NN only activates after this many samples
ML_CONFIDENCE_THRESHOLD = 0.80        # NN must beat this to override the rules
WEIGHTS_FILE            = 'nn_weights.npz'
TRAINING_FILE           = 'nn_training_data.json'

# ── Sorting logic ─────────────────────────────────────────────────────────────
#  Red/Green/Blue CUBES of normal size → matching colour bin
#  Everything else                     → side bin

# ── Connection ────────────────────────────────────────────────────────────────
client = RemoteAPIClient()
sim    = client.getObject('sim')

armjoint1, armjoint2, armjoint3, armjoint4, armjoint5, armjoint6 = (
    sim.getObject('/PArm/joint1'),
    sim.getObject('/PArm/joint1/joint2'),
    sim.getObject('/PArm/joint1/joint2/joint3'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4/joint5'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4/joint5/joint6'),
)

fingerjoint1 = sim.getObject('/PGripStraight/motor')
conveyor     = sim.getObject('/efficientConveyor')
sensor       = sim.getObject('/proximitySensor')
gripperTip   = sim.getObject('/PGripStraight/connector')

# ── Spawn settings ────────────────────────────────────────────────────────────
SPAWN_X      = -2.5
SPAWN_Y      =  0.0
SPAWN_Z      =  0.80
SHAPE_TYPES  = ['cube', 'cylinder', 'sphere']
COLOURS      = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
COLOUR_MAP   = {
    'red':    [1, 0, 0],   'green':  [0, 1, 0],   'blue':   [0, 0, 1],
    'yellow': [1, 1, 0],   'orange': [1, 0.5, 0], 'purple': [0.5, 0, 1],
}
spawned_objects = []

# ── Size boundaries (metres) ──────────────────────────────────────────────────
NORMAL_SIZE_MIN = 0.03
NORMAL_SIZE_MAX = 0.09

# ═══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORK
#  Input  : 6 features  [r, g, b, dim_x, dim_y, dim_z]
#  Output : 4 classes   ['red', 'green', 'blue', 'side']
#
#  Why these features?
#    - RGB values directly encode what the rule classifier checks for colour.
#    - Raw dimensions let the NN learn size + shape simultaneously.
#      A cube has roughly equal x/y/z; a cylinder will have one axis much
#      longer. The NN can discover this without being told explicitly.
#
#  In a new environment the arm bootstraps using the rule classifier as its
#  teacher. Once it has MIN_SAMPLES_BEFORE_ML examples it starts predicting
#  independently. Disagreements between the rule and the NN are still recorded
#  so the NN keeps improving. If the environment changes (different lighting
#  changes effective RGB readings, different object library) the NN adapts
#  while the rules remain as a safety net.
# ═══════════════════════════════════════════════════════════════════════════════
class SortingNN:
    CLASSES = ['red', 'green', 'blue', 'side']
    N_IN    = 6
    N_HID   = 24
    N_OUT   = 4

    def __init__(self, lr=0.03):
        self.lr            = lr
        self.W1            = np.random.randn(self.N_IN,  self.N_HID) * 0.1
        self.b1            = np.zeros(self.N_HID)
        self.W2            = np.random.randn(self.N_HID, self.N_OUT) * 0.1
        self.b2            = np.zeros(self.N_OUT)
        self.training_data = []   # list of (features, label_str)
        self.trained_count = 0
        self.objects_seen  = 0    # incremented once per pick-and-place cycle

    # ── Activations ──────────────────────────────────────────────────────────
    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    # ── Forward pass ─────────────────────────────────────────────────────────
    def _forward(self, features):
        x     = np.array(features, dtype=float)
        h_pre = x @ self.W1 + self.b1
        h     = self._relu(h_pre)
        out   = self._softmax(h @ self.W2 + self.b2)
        return x, h_pre, h, out

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, features):
        _, _, _, out = self._forward(features)
        idx = out.argmax()
        return self.CLASSES[idx], float(out[idx]), out.tolist()

    # ── Single back-prop step ─────────────────────────────────────────────────
    def train_step(self, features, label):
        x, h_pre, h, out = self._forward(features)
        y        = np.zeros(self.N_OUT)
        y[self.CLASSES.index(label)] = 1.0
        d_out    = out - y
        self.W2 -= self.lr * np.outer(h, d_out)
        self.b2 -= self.lr * d_out
        d_h      = (d_out @ self.W2.T) * (h_pre > 0)
        self.W1 -= self.lr * np.outer(x, d_h)
        self.b1 -= self.lr * d_h
        self.trained_count += 1

    # ── Experience replay ─────────────────────────────────────────────────────
    def replay(self, n=32):
        if len(self.training_data) < 4:
            return
        batch = random.sample(self.training_data, min(n, len(self.training_data)))
        for f, l in batch:
            self.train_step(f, l)

    # ── Add to replay buffer ──────────────────────────────────────────────────
    def remember(self, features, label):
        self.training_data.append((list(features), label))
        if len(self.training_data) > 2000:
            self.training_data.pop(0)

    # ── Save weights + training data to disk ──────────────────────────────────
    def save(self, weights_path=WEIGHTS_FILE, data_path=TRAINING_FILE):
        np.savez(
            weights_path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            trained_count=np.array([self.trained_count]),
            objects_seen=np.array([self.objects_seen]),
        )
        with open(data_path, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        print(f"  [NN] Saved → {weights_path}  "
              f"({self.trained_count} train steps, "
              f"{len(self.training_data)} samples, "
              f"{self.objects_seen} objects seen)")

    # ── Load weights + training data from disk ────────────────────────────────
    def load(self, weights_path=WEIGHTS_FILE, data_path=TRAINING_FILE):
        loaded_weights = False
        loaded_data    = False

        if os.path.exists(weights_path):
            try:
                d = np.load(weights_path)
                # Validate shape before accepting — guards against loading
                # weights from a different architecture
                if (d['W1'].shape == (self.N_IN, self.N_HID) and
                        d['W2'].shape == (self.N_HID, self.N_OUT)):
                    self.W1            = d['W1']
                    self.b1            = d['b1']
                    self.W2            = d['W2']
                    self.b2            = d['b2']
                    self.trained_count = int(d['trained_count'][0])
                    self.objects_seen  = int(d.get('objects_seen',
                                                   np.array([0]))[0])
                    loaded_weights = True
                    print(f"  [NN] Loaded weights from '{weights_path}'  "
                          f"(trained_count={self.trained_count}, "
                          f"objects_seen={self.objects_seen})")
                else:
                    print(f"  [NN] WARNING: '{weights_path}' has wrong shape — "
                          f"ignoring (expected W1={self.N_IN}x{self.N_HID}, "
                          f"W2={self.N_HID}x{self.N_OUT})")
            except Exception as e:
                print(f"  [NN] WARNING: Could not load weights — {e}")
        else:
            print(f"  [NN] No saved weights found at '{weights_path}' — starting fresh")

        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    raw = json.load(f)
                parsed = []
                for item in raw:
                    if (isinstance(item, list) and len(item) == 2
                            and isinstance(item[0], list)
                            and isinstance(item[1], str)
                            and item[1] in self.CLASSES):
                        parsed.append((item[0], item[1]))
                self.training_data = parsed
                loaded_data = True
                print(f"  [NN] Loaded {len(self.training_data)} training samples "
                      f"from '{data_path}'")
            except Exception as e:
                print(f"  [NN] WARNING: Could not load training data — {e}")
        else:
            print(f"  [NN] No saved training data found at '{data_path}' — starting fresh")

        return loaded_weights, loaded_data

    # ── Pretty status summary ─────────────────────────────────────────────────
    def status(self):
        active = (USE_ML and self.trained_count >= MIN_SAMPLES_BEFORE_ML)
        return (f"trained_count={self.trained_count}  "
                f"buffer={len(self.training_data)}  "
                f"objects_seen={self.objects_seen}  "
                f"active={'YES' if active else f'NO (need {MIN_SAMPLES_BEFORE_ML})'}")


# ── Initialise model ──────────────────────────────────────────────────────────
nn = SortingNN(lr=0.03)
if USE_ML and LOAD_MODEL:
    print("\n[NN] Loading saved model...")
    nn.load()

# ═══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED CLASSIFIERS  (always run; act as teacher for the NN)
# ═══════════════════════════════════════════════════════════════════════════════

def get_dims(handle):
    """Returns (x, y, z) bounding-box dimensions or (None, None, None)."""
    try:
        x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_x) \
          - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_x)
        y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_y) \
          - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_y)
        z = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z) \
          - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z)
        return x, y, z
    except Exception as e:
        print(f"  [SIZE] Error: {e}")
        return None, None, None

def get_rgb(handle):
    """Returns (r, g, b) float values or (None, None, None)."""
    try:
        _, colour = sim.getShapeColor(handle, None,
                                      sim.colorcomponent_ambient_diffuse)
        return float(colour[0]), float(colour[1]), float(colour[2])
    except Exception as e:
        print(f"  [COLOUR] Error: {e}")
        return None, None, None

def rule_classify(handle):
    """
    Deterministic rule classifier — the ground truth teacher.
    Returns one of: 'red', 'green', 'blue', 'side'
    """
    x, y, z = get_dims(handle)
    if None in (x, y, z):
        return 'side', None

    max_dim = max(x, y, z)
    min_dim = min(x, y, z)

    # Size gate
    if not (NORMAL_SIZE_MIN <= max_dim <= NORMAL_SIZE_MAX):
        print(f"  [RULE] SIDE — size {max_dim:.3f}m out of range")
        return 'side', None

    # Shape gate — cube has max/min ratio < 1.3
    ratio = max_dim / min_dim if min_dim > 0 else 99
    if ratio >= 1.3:
        print(f"  [RULE] SIDE — not a cube (ratio={ratio:.2f})")
        return 'side', None

    # Colour gate
    r, g, b = get_rgb(handle)
    if None in (r, g, b):
        return 'side', None

    print(f"  [RULE] dims={x:.3f}x{y:.3f}x{z:.3f}  ratio={ratio:.2f}  "
          f"R={r:.2f} G={g:.2f} B={b:.2f}")

    if r > 0.5 and g < 0.4 and b < 0.4:
        label = 'red'
    elif g > 0.5 and r < 0.4 and b < 0.4:
        label = 'green'
    elif b > 0.5 and r < 0.4 and g < 0.4:
        label = 'blue'
    else:
        label = 'side'

    features = [r, g, b, x, y, z]
    return label, features

# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED CLASSIFIER + ML INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify(handle):
    """
    1. Always runs rule_classify() to get ground-truth label + features.
    2. If USE_ML and enough training data, also runs the NN.
       - If NN agrees with rule:  use NN result (confidence shown).
       - If NN disagrees:         use rule result; log disagreement so NN learns.
       - If NN confidence low:    fall back to rule result.
    3. Trains and remembers regardless of which result is used.
    Returns: final_label (str)
    """
    rule_label, features = rule_classify(handle)

    if features is None:
        return rule_label

    nn_active = USE_ML and nn.trained_count >= MIN_SAMPLES_BEFORE_ML

    if nn_active:
        nn_label, confidence, probs = nn.predict(features)
        prob_str = '  '.join(
            f"{SortingNN.CLASSES[i]}={probs[i]:.2f}"
            for i in range(len(SortingNN.CLASSES))
        )
        print(f"  [NN]   prediction={nn_label}  conf={confidence:.2f}  [{prob_str}]")

        if confidence >= ML_CONFIDENCE_THRESHOLD:
            if nn_label == rule_label:
                print(f"  [NN]   ✓ agrees with rule → {nn_label.upper()}")
                final = nn_label
            else:
                # Disagreement: trust the rules, teach the NN
                print(f"  [NN]   ✗ disagrees (NN={nn_label}, rule={rule_label}) "
                      f"→ using RULE, correcting NN")
                final = rule_label
        else:
            print(f"  [NN]   confidence too low ({confidence:.2f} < "
                  f"{ML_CONFIDENCE_THRESHOLD}) → using RULE")
            final = rule_label
    else:
        if USE_ML:
            remaining = MIN_SAMPLES_BEFORE_ML - nn.trained_count
            print(f"  [NN]   not active yet — need {remaining} more training steps")
        final = rule_label

    # Always train on the rule label (rule is always the teacher)
    if USE_ML:
        nn.train_step(features, rule_label)
        nn.remember(features, rule_label)
        nn.replay(n=32)

    print(f"  ══ FINAL: {final.upper()} ══  {nn.status()}")
    return final

# ═══════════════════════════════════════════════════════════════════════════════
#  SPAWN Section
# ═══════════════════════════════════════════════════════════════════════════════

def spawn_object():
    shape_type = random.choice(SHAPE_TYPES)
    dim        = random.uniform(0.04, 0.085)
    y_offset   = random.uniform(-0.03, 0.03)
    colour     = random.choice(COLOURS)

    try:
        if shape_type == 'cube':
            handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid,
                                              [dim, dim, dim], 1)
        elif shape_type == 'cylinder':
            height = dim * random.uniform(1.5, 2.5)
            handle = sim.createPrimitiveShape(sim.primitiveshape_cylinder,
                                              [dim, dim, height], 1)
        else:
            handle = sim.createPrimitiveShape(sim.primitiveshape_spheroid,
                                              [dim, dim, dim], 1)

        sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse,
                          COLOUR_MAP[colour])
        sim.setShapeMass(handle, 0.1)
        sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, 1)
        sim.setObjectInt32Param(handle, sim.shapeintparam_static, 0)
        sim.resetDynamicObject(handle)
        sim.setObjectPosition(handle, -1,
                              [SPAWN_X, SPAWN_Y + y_offset, SPAWN_Z])
        spawned_objects.append(handle)
        print(f"  [SPAWN] {shape_type}  size={dim:.3f}m  colour={colour}  "
              f"handle={handle}")
        return handle

    except Exception as e:
        print(f"  [SPAWN] Error: {e}")
        return None

def steps(n):
    for _ in range(n):
        sim.step()

def startingpostion():
    for joint in (armjoint1, armjoint2, armjoint3,
                  armjoint4, armjoint5, armjoint6):
        sim.setJointTargetPosition(joint, 0)
    steps(400)

def detection(n):
    sim.setJointTargetPosition(armjoint2, 1.4)
    sim.setJointTargetPosition(armjoint3, 1.77)
    sim.setJointTargetPosition(armjoint5, -1.4)
    steps(200)
    sim.setJointTargetPosition(fingerjoint1, -0.044)
    steps(80)
    sim.setObjectParent(n, gripperTip, True)

def scan_position():
    sim.setJointTargetPosition(armjoint1, 0)
    sim.setJointTargetPosition(armjoint2, 0.8)
    steps(300)

def _drop(n, joint1_angle):
    sim.setJointTargetPosition(armjoint1, joint1_angle)
    steps(200)
    sim.setJointTargetPosition(armjoint5, -1.5)
    steps(500)
    sim.setJointTargetPosition(armjoint2, 1)
    steps(200)
    sim.setJointTargetPosition(fingerjoint1, 0)
    sim.setObjectParent(n, -1, True)
    steps(200)

def drop_red(n):   _drop(n,  1.0)
def drop_green(n): _drop(n,  2.2)
def drop_blue(n):  _drop(n,  1.6)
def drop_side(n):  _drop(n, -1.0)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

sim.setStepping(True)
sim.startSimulation()

print("\n" + "=" * 65)
print("  Colour-Cube Sorting Arm")
print(f"  ML:          {'ENABLED' if USE_ML else 'DISABLED'}")
if USE_ML:
    print(f"  Load model:  {'YES' if LOAD_MODEL else 'NO'}")
    print(f"  Activates:   after {MIN_SAMPLES_BEFORE_ML} training samples")
    print(f"  Confidence:  must exceed {ML_CONFIDENCE_THRESHOLD} to override rules")
    print(f"  Save every:  {SAVE_EVERY_N} objects")
    print(f"  Files:       {WEIGHTS_FILE}  |  {TRAINING_FILE}")
print("  Sorting:     Red/Green/Blue CUBES → colour bin | rest → side")
print("  Keys:        Q = stop  |  S = spawn object")
print("=" * 65 + "\n")

for joint in (armjoint1, armjoint2, armjoint3, armjoint5):
    sim.setObjectFloatParam(joint, sim.jointfloatparam_maxvel,   0.1)
    sim.setObjectFloatParam(joint, sim.jointfloatparam_maxaccel, 0.1)

sim.setBufferProperty(conveyor, 'customData.__ctrl__',
                      sim.packTable({'vel': 0.017}))

grasped       = False
objects_sorted = 0
spawn_object()

try:
    while True:
        if keyboard.is_pressed('q'):
            print("\nQ pressed — stopping...")
            break

        if keyboard.is_pressed('s'):
            spawn_object()
            steps(50)

        detected, distance, detectedObjloc, detectedObjHandle, two22 = \
            sim.readProximitySensor(sensor)

        if detected and not grasped:
            grasped = True
            print("─" * 65)
            print(f"Object detected (handle={detectedObjHandle})")
            sim.setBufferProperty(conveyor, 'customData.__ctrl__',
                                  sim.packTable({'vel': 0.0}))

            # 1. Pick up
            detection(detectedObjHandle)
            steps(400)

            # 2. Scan position
            scan_position()

            # 3. Classify (rules + optional NN)
            category = classify(detectedObjHandle)

            # 4. Track objects seen (used for periodic save)
            if USE_ML:
                nn.objects_seen += 1

            # 5. Return home
            startingpostion()
            steps(400)

            # 6. Drop to correct bin
            if category == 'red':
                drop_red(detectedObjHandle)
            elif category == 'green':
                drop_green(detectedObjHandle)
            elif category == 'blue':
                drop_blue(detectedObjHandle)
            else:
                drop_side(detectedObjHandle)

            steps(400)
            objects_sorted += 1

            # 7. Periodic save
            if USE_ML and objects_sorted % SAVE_EVERY_N == 0:
                print(f"  [NN] Periodic save (every {SAVE_EVERY_N} objects)...")
                nn.save()

            # 8. Home, restart belt, next object
            startingpostion()
            steps(400)
            sim.setBufferProperty(conveyor, 'customData.__ctrl__',
                                  sim.packTable({'vel': 0.017}))
            spawn_object()
            grasped = False

        sim.step()

finally:
    if USE_ML:
        print("\n[NN] Saving model before exit...")
        nn.save()
    sim.stopSimulation()
    print(f"Simulation stopped. Total objects sorted: {objects_sorted}")