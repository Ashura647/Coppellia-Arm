from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

# Full path to the table model
model_path = r"C:\Program Files\CoppeliaRobotics\CoppeliaSim\models\furniture\table.ttm"

# Load the model
table = sim.loadModel(model_path)

# Move it up slightly
sim.setObjectPosition(table, -1, [0, 0, 0])

print("Table loaded successfully.")