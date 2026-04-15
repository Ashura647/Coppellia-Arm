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
conveyor = sim.getObject('/efficientConveyor')
sensor = sim.getObject('/proximitySensor')
cube1 = sim.getObject('/Cuboid')
gripperTip= sim.getObject('/PGripStraight/connector')

def startingpostion():
    sim.setJointTargetPosition(armjoint1, 0)
    sim.setJointTargetPosition(armjoint2, 0)
    sim.setJointTargetPosition(armjoint3, 0)
    sim.setJointTargetPosition(armjoint4, 0)
    sim.setJointTargetPosition(armjoint5, 0)
    sim.setJointTargetPosition(armjoint6, 0)
    #sim.setJointTargetPosition(fingerjoint1, 0)
    for _ in range(400):
        sim.step()

def detection():
    sim.setJointTargetPosition(armjoint2, 1.4)
    sim.setJointTargetPosition(armjoint3, 1.77)
    sim.setJointTargetPosition(armjoint5, -1.4)
    for _ in range(200):
        sim.step()

    sim.setJointTargetPosition(fingerjoint1, -0.044)

    for _ in range(80):
        sim.step()

    sim.setObjectParent(cube1, gripperTip, True)
   

    
 
sim.setStepping(True)
sim.startSimulation()

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

sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.017}))

grasped = False
while True:
    detected, distance, detectedObj, _, _ = sim.readProximitySensor(sensor)
    
    
    if detected and not grasped:
        print("Object detected! Stopping conveyor")

        sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.0}))
        detection()
        for _ in range(400):
            sim.step()
        print(sim.getJointPosition(armjoint3))
        startingpostion()
        for _ in range(800):
            sim.step()
        break
        
    sim.step()


sim.stopSimulation()



