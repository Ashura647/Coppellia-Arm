from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient()
sim = client.getObject('sim')

armjoint1 = sim.getObject('/PArm/joint1')
armjoint2 = sim.getObject('/PArm/joint1/joint2')
gripperTip = sim.getObject('/PGripStraight/connector')
visionSensor = sim.getObject('/sphericalVisionRGBAndDepth/sensorRGB')  # ← add this

sim.setStepping(True)
sim.startSimulation()

sim.setJointTargetPosition(armjoint1, 0)
sim.setJointTargetPosition(armjoint2, 0.8)

for _ in range(300):
    sim.step()

x, y, z = sim.getObjectPosition(gripperTip, -1)
print(f"joint2=0.8 → gripper at x={x:.3f}, y={y:.3f}, z={z:.3f}")

result = sim.readVisionSensor(visionSensor)
print(f"Sensor result at scan position: {result}")

sim.stopSimulation()