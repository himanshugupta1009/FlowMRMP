import pybullet as p
import pybullet_data
import datetime
import numpy as np
import quaternion

import sys
import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, '.')
sys.path.insert(0, '../')
print(sys.path)

import importlib  
pybullet_utils = importlib.import_module("pybullet-planning.pybullet_tools.utils")

""" 
A file I (Paul) use to test pybullet functionality, quaternions, etc. 
"""


# Creates a pybullet world and a visualizer for it
pybullet_utils.connect(use_gui=True)
pybullet_utils.set_camera_pose(camera_point=[1, -1.5, 1], target_point=pybullet_utils.unit_point()) # Sets the camera's position

base_po = None 
base_v = None
joint_states = [] # joint states for each body

TURTLEBOT_NH_URDF = pybullet_utils.join_paths(pybullet_utils.MODEL_DIRECTORY, 'turtlebot/turtlebot.urdf')


from edge_bundle import EdgeBundleTraj
import numpy as np
edge_bundle_file_location = 'edge_bundles/eb_pb_turtle_speed_20_edges-10000.npz' 
data = np.load(edge_bundle_file_location, allow_pickle=True)
eb_turtle = EdgeBundleTraj(data, fix_num_edges=5)

def hex_to_rgba(hex_color):
        col_val = [int(hex_color[i:i+2], 16)/(2**8 - 1) for i in (0, 2, 4)] + [1]
        return col_val

def bot_to_hex(hex_color, bot):
        col_val = [int(hex_color[i:i+2], 16)/(2**8 - 1) for i in (0, 2, 4)] + [1]
        for j in range (-1, p.getNumJoints(bot)):
                p.changeVisualShape(bot, j, rgbaColor=col_val)        

def mult_quaternion(qa, qb):
        npa = np.quaternion(qa[3], qa[0], qa[1], qa[2])
        npb = np.quaternion(qb[3], qb[0], qb[1], qb[2])
        res = npa * npb 
        return (res.x, res.y, res.z, res.w)

def rotate_vector(quat, vector):
        np_quat = np.quaternion(quat[3], quat[0], quat[1], quat[2])
        np_quat_inv = np.quaternion(quat[3], -quat[0], -quat[1], -quat[2])
        vec_quat = np.quaternion(0., vector[0], vector[1], vector[2])
        res = np_quat * vec_quat * np_quat_inv
        return (res.x, res.y, res.z) 
        

def point_translate_function(base_point, edge_point):
        rot = rotate_vector(base_point[1], edge_point[0])
        x = base_point[0][0] + rot[0]
        y = base_point[0][1] + rot[1]
        z = rot[2]

        return ((x, y, z), mult_quaternion(base_point[1], edge_point[1]))

with pybullet_utils.LockRenderer(): # Temporarily prevents the renderer from updating for improved loading efficiency
        with pybullet_utils.HideOutput(): # Temporarily suppresses pybullet output
                offset = [0,0,0.2]

                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                plane = p.loadURDF("plane.urdf")

                print("Loading Turtlebot from " + pybullet_utils.TURTLEBOT_URDF)
                turtle = p.loadURDF(TURTLEBOT_NH_URDF, [-1, -1, 0.2])
                bot_to_hex("380282", turtle)
                z = pybullet_utils.stable_z(turtle, plane)
                print("\n\n\n" + str(z) + "\n\n\n")

                turtle2 = p.loadURDF(pybullet_utils.TURTLEBOT_URDF, [0,0,0.2])

                base_po = p.getBasePositionAndOrientation(turtle)
                base_v = p.getBaseVelocity(turtle)
                joint_states = [p.getJointState(turtle,j) for j in range(p.getNumJoints(turtle))]

                # set up obstacles
                obs = pybullet_utils.create_box(w=1, l=1, h=0.1, color=hex_to_rgba("04d8b2"), collision=False)
                pybullet_utils.set_point(obs, [0,3,0.1 / 2.]);
                pybullet_utils.set_euler(obs, [0, 0, 0])
                pybullet_utils.set_static(obs)
                p.setCollisionFilterPair(turtle, obs, -1, -1, 1)


for j in range (p.getNumJoints(turtle)):
        print(p.getJointInfo(turtle,j))

for j in range (p.getNumJoints(turtle2)):
        print(p.getJointInfo(turtle2,j))

p.setRealTimeSimulation(0)
p.setTimeStep(1 / 240)

forward=0
turn=0
steperator = 0
while (1):
        p.setGravity(0,0,-10)
        # time.sleep(1./240.)
        leftWheelVelocity=0
        rightWheelVelocity=0
        speed=1
        keys = p.getKeyboardEvents()
	
        for k,v in keys.items():
                if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        turn = -0.5
                if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
                        turn = 0
                if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        turn = 0.5
                if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
                        turn = 0

                if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        forward=1
                if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
                        forward=0
                if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        forward=-1
                if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                        forward=0

                if k == 65309 and (v&p.KEY_WAS_RELEASED):
                        # set_joint_position(turtle, 0, 0) # Sets the current value of the x joint
                        # set_joint_position(turtle, 1, -1) # Sets the current value of the y joint       
                        # set_joint_position(turtle, 2, 0) 
                        # wait_for_duration(1)
                        p.resetBasePositionAndOrientation(turtle, *base_po)
                        p.resetBaseVelocity(turtle, *base_v)
                        print("Rst Pos: " + str(base_po))
                        print("Rst Vel: " + str(base_v))
                        # for j in range(p.getNumJoints(turtle)):
                        #         p.resetJointState(turtle, j, *joint_states[j][:2])

                if k == 115 and (v&p.KEY_WAS_RELEASED):
                        base_po = p.getBasePositionAndOrientation(turtle)
                        # base_v = p.getBaseVelocity(turtle)
                        joint_states = [p.getJointState(turtle,j) for j in range(p.getNumJoints(turtle))]
                        print("Saved Pos: " + str(base_po))
                        print("Saved Vel: " + str(base_v))

                if k == 112 and (v&p.KEY_WAS_RELEASED):
                        # # print("Pos: " + str(p.getBasePositionAndOrientation(turtle)))
                        # # print("Vel: " + str(p.getBaseVelocity(turtle)))
                        curpos = p.getBasePositionAndOrientation(turtle)
                        curpos2 = p.getBasePositionAndOrientation(turtle2)
                        # newpos = (curpos[0], mult_quaternion(curpos[1], base_po[1]))
                        # p.resetBasePositionAndOrientation(turtle, *newpos)
                        # p.resetBaseVelocity(turtle, *base_v)
                        for i in range(eb_turtle.num_edges):
                                traj = eb_turtle.get_trajectory(i)
                                cont, _, _ = eb_turtle.get_edge(i)
                                print("Control: " + str(cont))
                                for j in range(len(traj)):
                                        trans_pos = point_translate_function(curpos, traj[j])
                                        p.resetBasePositionAndOrientation(turtle, *trans_pos)
                                        p.resetBaseVelocity(turtle, *base_v)

                                        trans_pos = point_translate_function(curpos2, traj[j])
                                        p.resetBasePositionAndOrientation(turtle2, *trans_pos)
                                        p.resetBaseVelocity(turtle2, *base_v)

                                        pybullet_utils.wait_for_duration(1.0)
                                print("Edge " + str(i) + " done!")
                                pybullet_utils.wait_for_duration(3.0)     
                                p.resetBasePositionAndOrientation(turtle, *curpos)    
                                p.resetBasePositionAndOrientation(turtle2, *curpos2)                

        # if steperator < 2000:
        #         forward = 1
        # elif steperator < 4000:
        #         forward = 0
        #         turn  = -0.5
        # else:
        #         forward = 0
        #         turn = 0
        #         steperator = 0

        rightWheelVelocity += (forward+turn)*speed
        leftWheelVelocity += (forward-turn)*speed
	
        p.setJointMotorControl2(turtle,0,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
        p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)

        if pybullet_utils.pairwise_collision(turtle, obs):
                print("Hit something ya dummy!!! " + str(datetime.datetime.now().time()))
        
        if pybullet_utils.pairwise_collision(turtle, turtle2):
                print("Hit another turtle!!! The Turltemanity!!!!!" + str(datetime.datetime.now().time()))

        p.stepSimulation()
