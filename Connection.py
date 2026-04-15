from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
client = RemoteAPIClient()
sim = client.getObject('sim')

armjoint1,armjoint2,armjoint3,armjoint4,armjoint5,armjoint6=( 
    sim.getObject('/PArm/joint1'),
    sim.getObject('/PArm/joint1/joint2'),
    sim.getObject('/PArm/joint1/joint2/joint3'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4/joint5'),
    sim.getObject('/PArm/joint1/joint2/joint3/joint4/joint5/joint6'),
)

fingerjoint1 = sim.getObject('/PGripStraight/motor')
conveyor_joint = sim.getObject('/efficientConveyor')
sensor = sim.getObject('/proximitySensor')

sim.startSimulation()

# while True:
#     #Read proximity sensor: result(0/1), distance, detectedPoint, detectedObjectHandle, normalVector
#     res, distance, detectedPoint, detectedObjectHandle, normalVector = sim.readProximitySensor(sensor)
#     if res > 0:
#         # Object detected: Stop conveyor
#         sim.setJointTargetVelocity(conveyor_joint, 0)
#     else:
#         # No object: Run conveyor
#         sim.setJointTargetVelocity(conveyor_joint, 2)
    

#starting param
sim.setObjectFloatParam(armjoint1, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint1, sim.jointfloatparam_maxaccel, 0.1)
sim.setObjectFloatParam(armjoint5, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint5, sim.jointfloatparam_maxaccel, 0.1)
sim.setObjectFloatParam(armjoint3, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint3, sim.jointfloatparam_maxaccel, 0.1)
sim.setObjectFloatParam(armjoint5, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint5, sim.jointfloatparam_maxaccel, 0.1)
sim.setObjectFloatParam(armjoint2, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint2, sim.jointfloatparam_maxaccel, 0.1)



sim.setJointTargetPosition(armjoint2, 1.3)
sim.setJointTargetPosition(armjoint3, 2)
sim.setJointTargetPosition(armjoint5, -1.2)
time.sleep(20)
sim.setJointTargetPosition(fingerjoint1, -0.044)
sim.setObjectFloatParam(armjoint3, sim.jointfloatparam_maxvel, 0.1)
sim.setObjectFloatParam(armjoint3, sim.jointfloatparam_maxaccel, 0.1)
sim.setJointTargetPosition(armjoint2, 0)

time.sleep(30)
sim.stopSimulation()



