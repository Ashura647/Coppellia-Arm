from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print("Connecting to CoppeliaSim...")

client = RemoteAPIClient()
sim = client.getObject('sim')

print("Connected successfully!")

# Start simulation
sim.startSimulation()

print("Simulation started")

sim.stopSimulation()
print("Simulation stopped")
