from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
client = RemoteAPIClient()
sim = client.getObject('sim')

joint1= sim.getObject('/PArm/joint')
sim.startSimulation()

time.sleep(3)

sim.setObjectFloatParam(joint1, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(joint1, sim.jointfloatparam_maxaccel, 0.1)
sim.setJointTargetPosition(joint1, 1.57)


time.sleep(10)
sim.stopSimulation()



