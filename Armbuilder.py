from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import random
import math
import numpy as np
import keyboard

# ── Neural network (simple 3-layer MLP, trains live) ─────────────────────────
class LiveNN:
    CLASSES = ['small_cube','large_cube','small_cyl','large_cyl',
               'small_sphere','large_sphere','defect']

    def __init__(self, lr=0.05):
        self.lr = lr
        self.W1 = np.random.randn(5, 16) * 0.1
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(16, 7) * 0.1
        self.b2 = np.zeros(7)
        self.training_data = []
        self.trained_count  = 0

    def _relu(self, x):   return np.maximum(0, x)
    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def predict(self, features):
        x   = np.array(features, dtype=float)
        h   = self._relu(x @ self.W1 + self.b1)
        out = self._softmax(h @ self.W2 + self.b2)
        return self.CLASSES[out.argmax()], float(out.max())

    def train_step(self, features, label):
        x  = np.array(features, dtype=float)
        y  = np.zeros(7); y[self.CLASSES.index(label)] = 1.0
        h_pre = x @ self.W1 + self.b1
        h     = self._relu(h_pre)
        out   = self._softmax(h @ self.W2 + self.b2)
        d_out = out - y
        self.W2 -= self.lr * np.outer(h, d_out)
        self.b2 -= self.lr * d_out
        d_h   = (d_out @ self.W2.T) * (h_pre > 0)
        self.W1 -= self.lr * np.outer(x, d_h)
        self.b1 -= self.lr * d_h
        self.trained_count += 1

    def replay(self, n=32):
        if len(self.training_data) < 4:
            return
        batch = random.sample(self.training_data, min(n, len(self.training_data)))
        for f, l in batch:
            self.train_step(f, l)

    def remember(self, features, label):
        self.training_data.append((features, label))
        if len(self.training_data) > 500:
            self.training_data.pop(0)

# ── Connection ────────────────────────────────────────────────────────────────
client = RemoteAPIClient()
sim    = client.getObject('sim')

armjoint1,armjoint2,armjoint3,armjoint4,armjoint5,armjoint6 = (
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

# ── Bin angles (joint1 rotation to reach each bin) ───────────────────────────
BIN_ANGLES = {
    'small_cube':   0.8,
    'large_cube':   1.1,
    'small_cyl':    1.4,
    'large_cyl':    1.7,
    'small_sphere': 2.0,
    'large_sphere': 2.3,
    'defect':      -1.0,
}

# ── Size boundaries (metres) ──────────────────────────────────────────────────
SMALL_MAX  = 0.055
LARGE_MIN  = 0.056
DEFECT_MAX = 0.12
DEFECT_MIN = 0.02

# ── Spawn settings ────────────────────────────────────────────────────────────
SPAWN_X    = -2.5
SPAWN_Y    =  0.0
SPAWN_Z    =  0.80

SHAPE_TYPES    = ['cube', 'cylinder', 'sphere']
spawned_objects = []
nn = LiveNN(lr=0.05)

# ── Feature extraction ────────────────────────────────────────────────────────
def get_features(handle):
    try:
        x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_x) \
          - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_x)
        y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_y) \
          - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_y)
        z = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z) \
          - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z)
        xy = x / y if y > 0 else 1.0
        xz = x / z if z > 0 else 1.0
        return [x, y, z, xy, xz]
    except Exception as e:
        print(f"  [FEAT] Error: {e}")
        return None

# ── Rule-based classifier ─────────────────────────────────────────────────────
def rule_classify(features):
    x, y, z, xy, xz = features
    max_dim = max(x, y, z)
    min_dim = min(x, y, z)
    ratio   = max_dim / min_dim if min_dim > 0 else 1.0

    if max_dim > DEFECT_MAX or max_dim < DEFECT_MIN:
        return 'defect'

    if ratio > 1.4:
        shape = 'cyl'
    elif abs(xy - 1.0) < 0.15 and abs(xz - 1.0) < 0.15:
        shape = 'cube'
    else:
        shape = 'sphere'

    size = 'small' if max_dim <= SMALL_MAX else 'large'
    return f"{size}_{shape}"

# ── Combined classifier ───────────────────────────────────────────────────────
def classify(handle):
    features = get_features(handle)
    if features is None:
        return 'defect', features

    rule_label = rule_classify(features)
    print(f"  [RULE] → {rule_label}  (dims: {features[0]:.3f} x {features[1]:.3f} x {features[2]:.3f})")

    if nn.trained_count >= 10:
        nn_label, confidence = nn.predict(features)
        print(f"  [NN]   → {nn_label}  (confidence: {confidence:.2f}, trained on {nn.trained_count} samples)")
        final = nn_label if confidence > 0.75 else rule_label
    else:
        final = rule_label
        print(f"  [NN]   → not enough data yet ({nn.trained_count}/10 samples)")

    return final, features

# ── Spawn a random object on the belt ────────────────────────────────────────
def spawn_object():
    shape_type = random.choice(SHAPE_TYPES)

    size_roll = random.random()
    if size_roll < 0.1:
        dim = random.uniform(0.005, DEFECT_MIN - 0.001)
    elif size_roll > 0.9:
        dim = random.uniform(DEFECT_MAX + 0.001, 0.15)
    elif size_roll < 0.5:
        dim = random.uniform(0.025, SMALL_MAX)
    else:
        dim = random.uniform(LARGE_MIN, 0.10)

    y_offset = random.uniform(-0.03, 0.03)

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

        # Random colour
        colour = random.choice(['red','green','blue','yellow','orange','purple'])
        colour_map = {
            'red':    [1,0,0], 'green':  [0,1,0], 'blue':   [0,0,1],
            'yellow': [1,1,0], 'orange': [1,0.5,0], 'purple': [0.5,0,1],
        }
        sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse,
                          colour_map[colour])

        # Dynamic properties — must be set for conveyor to move the object
        sim.setShapeMass(handle, 0.1)
        sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, 1)
        sim.setObjectInt32Param(handle, sim.shapeintparam_static, 0)
        sim.resetDynamicObject(handle)

        # Place on belt
        sim.setObjectPosition(handle, -1, [SPAWN_X, SPAWN_Y + y_offset, SPAWN_Z])

        spawned_objects.append(handle)
        print(f"  [SPAWN] {shape_type} size={dim:.3f}m colour={colour} handle={handle}")
        return handle

    except Exception as e:
        print(f"  [SPAWN] Error: {e}")
        return None

# ── Arm motion ────────────────────────────────────────────────────────────────
def steps(n):
    for _ in range(n):
        sim.step()

def startingpostion():
    sim.setJointTargetPosition(armjoint1, 0)
    sim.setJointTargetPosition(armjoint2, 0)
    sim.setJointTargetPosition(armjoint3, 0)
    sim.setJointTargetPosition(armjoint4, 0)
    sim.setJointTargetPosition(armjoint5, 0)
    sim.setJointTargetPosition(armjoint6, 0)
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

def drop_to_bin(n, bin_label):
    angle = BIN_ANGLES.get(bin_label, BIN_ANGLES['defect'])
    sim.setJointTargetPosition(armjoint1, angle)
    steps(200)
    sim.setJointTargetPosition(armjoint5, -1.5)
    steps(500)
    sim.setJointTargetPosition(armjoint2, 1)
    steps(200)
    sim.setJointTargetPosition(fingerjoint1, 0)
    sim.setObjectParent(n, -1, True)
    steps(200)

# ── Main ──────────────────────────────────────────────────────────────────────
sim.setStepping(True)
sim.startSimulation()

print("=" * 50)
print("  Smart Sorting Arm — Live Learning Mode")
print("  Press Q to stop | Press S to spawn object")
print("=" * 50)

for joint in (armjoint1, armjoint2, armjoint3, armjoint5):
    sim.setObjectFloatParam(joint, sim.jointfloatparam_maxvel,   0.1)
    sim.setObjectFloatParam(joint, sim.jointfloatparam_maxaccel, 0.1)

sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.017}))

grasped    = False
step_count = 0

# Spawn first object
spawn_object()

while True:
    if keyboard.is_pressed('q'):
        print("Stopping simulation...")
        sim.stopSimulation()
        break

    if keyboard.is_pressed('s'):
        spawn_object()
        steps(50)

    detected, distance, detectedObjloc, detectedObjHandle, two22 = sim.readProximitySensor(sensor)

    if detected and not grasped:
        grasped = True
        print("─" * 50)
        print(f"Object detected (handle={detectedObjHandle}) — stopping conveyor")
        sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.0}))

        # 1. Pick up
        detection(detectedObjHandle)
        steps(400)

        # 2. Scan position
        print("  Moving to scan position...")
        scan_position()

        # 3. Classify
        print("  Classifying...")
        category, features = classify(detectedObjHandle)
        print(f"  → Final category: {category.upper()}")

        # 4. Train NN
        if features is not None:
            rule_label = rule_classify(features)
            nn.train_step(features, rule_label)
            nn.remember(features, rule_label)
            nn.replay(n=16)
            print(f"  [NN] Trained. Total samples: {nn.trained_count}")

        # 5. Return home
        startingpostion()
        steps(400)

        # 6. Drop to bin
        drop_to_bin(detectedObjHandle, category)
        steps(400)

        # 7. Home again
        startingpostion()
        steps(400)

        # 8. Restart belt
        sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.017}))

        # 9. Spawn next object
        spawn_object()

        grasped = False

    step_count += 1
    sim.step()

sim.stopSimulation()