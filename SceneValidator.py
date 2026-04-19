from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient()
sim = client.getObject('sim')

sim.setStepping(True)
sim.startSimulation()

# Check conveyor parts
conveyor = sim.getObject('/efficientConveyor')
x, y, z = sim.getObjectPosition(conveyor, -1)
print(f"Conveyor centre: x={x:.3f}, y={y:.3f}, z={z:.3f}")

# Check where existing cuboids spawn
for i in range(3):
    try:
        cube = sim.getObject(f'/Cuboid[{i}]')
        x, y, z = sim.getObjectPosition(cube, -1)
        print(f"Cuboid[{i}]: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    except:
        pass

sim.stopSimulation()