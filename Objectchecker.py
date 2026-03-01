from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

sim.startSimulation()

print("All objects in scene:\n")

objs = sim.getObjectsInTree(sim.handle_scene)
for o in objs:
    print(sim.getObjectAlias(o))

sim.stopSimulation()
