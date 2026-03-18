from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
client = RemoteAPIClient()
sim = client.getObject('sim')

armjoint1,armjoint2,armjoint3,armjoint4,armjoint5=( 
    sim.getObject('/PArm/joint1'),
    sim.getObject('/PArm/joint1/joint2'),
    sim.getObject('/PArm/joint1/joint2/joint3'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4/joint5'),
)


sim.startSimulation()

time.sleep(3)

sim.setObjectFloatParam(armjoint1, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint1, sim.jointfloatparam_maxaccel, 0.1)
sim.setJointTargetPosition(armjoint1, -0.5)

time.sleep(3)

sim.setObjectFloatParam(armjoint2, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint2, sim.jointfloatparam_maxaccel, 0.1)
sim.setJointTargetPosition(armjoint2, 1.4) 

time.sleep(3)

sim.setObjectFloatParam(armjoint3, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint3, sim.jointfloatparam_maxaccel, 0.1)
sim.setJointTargetPosition(armjoint3, 1)

time.sleep(3)





time.sleep(5)
sim.stopSimulation()



