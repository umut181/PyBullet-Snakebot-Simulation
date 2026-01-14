import pybullet as p
import pybullet_data
import time
import math
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import random
import pandas as pd
import os

''' Control Panel:

Define your trial number, select your task and motion gait.
Use "Z" and "X" keys to steer the robot.
Use "R" to lock the camera.

'''
trial_num = 1

#:::::::::::::::TASK SELECTION:::::::::::::::

TASK = "HIGH-FRICTION RAMP-STAIRS"
end_threshold_x = 16

'''
TASK = "RUBBLE"
end_threshold_x = 10
'''

#:::::::::::::::GAIT SELECTION:::::::::::::::

GAIT = "UNDULATION"  
final_force = 10
rot = 0
frictionSet = [0.3, 1.5, 0.1]
robot = "snakebot_mk1.urdf"

'''
GAIT = "SIDEWINDING"
final_force = 10
rot = math.pi/2
frictionSet = [0.3, 1.5, 0.1]
robot = "snakebot_mk2.urdf"
'''

''' Inching Gait is Still Under Development!

#GAIT = "INCHING"
#final_force = 2
#rot = 0
#frictionSet = [0.1, 1.5, 0.5]
#robot = "snakebot_mk1.urdf"
'''

#--- Data Collection ---------------------------------

# === Initalization Values ===

start_pos = None
last_pos = None


next_sample_time = 0
sampling_interval = 1/240  # seconds 
com_velocity_data = []

head_link_index = 11

total_energy = 0
total_dist = 0

link_indices = [-1,0,1,2,3,4,5,6,7,8,9,10,11]

force_sums = {link: 0.0 for link in link_indices}
step_count = 0

contact_fraction_sum = 0
contact_durations = {link: 0 for link in link_indices}
total_steps = 0


start_threshold_x = 2

anchoring_link_indices = [-1,12]
anchoring_force = -10

# === Manual Logging Control ===

logging_enabled = False
logging_start_time = None
logging_duration = 15 # seconds of logging after activation
debug_text_id = None  # will hold the ID of the on-screen debug text

# ==============

# === Functions ===
def create_wall(x, y, length, height=1.0, thickness=0.1, orientation = 0):
    
    orn = p.getQuaternionFromEuler([0,0,orientation])
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length / 2, thickness / 2, height / 2])
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=shape,
        basePosition=[x, y, height / 2],
        baseOrientation = orn
    )

def is_touching_ground(body_id, link_index):
    contacts = p.getContactPoints(bodyA=body_id, linkIndexA=link_index)
    return len(contacts) > 0


def follow_robot_camera(robot_id, head_link_index):
    # Get the position of the robot (or a link)
    link_state = p.getLinkState(robot_id, head_link_index)
    link_pos = link_state[0]

    # Set your preferred view parameters here:
    camera_distance = 3        # how far the camera is from the robot
    camera_yaw = 270           # left-right angle (0 = x+, 90 = y+)
    camera_pitch = -40         # up-down angle (0 = horizontal, -90 = top-down)

    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=link_pos
    )





def initialize_force_sums():
    return {link: 0.0 for link in link_indices}

def initialize_contact_durations():
    return {link: 0.0 for link in link_indices}

def update_contact_durations(robot_id, contact_durations):
    contact_points = p.getContactPoints(bodyA=robot_id)
    contacts_per_link = set()

    for point in contact_points:
        link_idx = point[3]
        contacts_per_link.add(link_idx)

    # Increment contact duration counters
    for link in contact_durations:
        if link in contacts_per_link:
            contact_durations[link] += 1

    return contact_durations

def update_contact_fraction(robot_id, contact_fraction_sum, total_links):
    contact_points = p.getContactPoints(bodyA=robot_id)
    contacts_per_link = set()

    for point in contact_points:
        link_idx = point[3]
        contacts_per_link.add(link_idx)

    # Calculate fraction of links in contact this timestep
    fraction_in_contact = len(contacts_per_link) / total_links

    # Accumulate this fraction over time
    contact_fraction_sum += fraction_in_contact

    return contact_fraction_sum


def update_force_sums(robot_id, link_indices, force_sums,):
    contacts = p.getContactPoints(robot_id)

    for contact in contacts:
        link_idx = contact[3]
        normal_force = contact[9]
        if link_idx in link_indices:
            force_sums[link_idx] = force_sums.get(link_idx, 0) + normal_force

    return force_sums

def update_steps(step_count):
    step_count +=1
    return step_count


def check_start_logging(head_x):
    global logging_enabled, logging_start_time, start_pos

    if head_x > start_threshold_x:
        if not logging_enabled:
            reset_logging_data()
            start_pos, _ = p.getBasePositionAndOrientation(snakebot)
        
            logging_enabled = True
            logging_start_time = time.time()
            print("🚀 Logging started!")

            # Add red on-screen message
            debug_text_id = p.addUserDebugText(
                "LOGGING ACTIVE",
                [0, 0, 1.5],
                textColorRGB=[1, 0, 0],
                textSize=2,
                lifeTime=0  # stays forever until manually removed
            )

def reset_logging_data():
    global force_sums, step_count, total_energy, total_dist, total_steps
    global contact_durations, com_velocity_data, prev_pos

    force_sums = initialize_force_sums()
    step_count = 0
    total_energy = 0.0
    total_dist = 0.0
    total_steps = 0
    contact_durations = initialize_contact_durations()
    com_velocity_data = []
    prev_pos = p.getBasePositionAndOrientation(snakebot)[0]

#-----------------------------------------------------


# ===== Simulation Setup =====
physicsClient = p.connect(p.GUI)  # or p.DIRECT for headless
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

''' DO NOT UNCOMMENT - This is to retrieve default plane for debugging
terrain_id = p.loadURDF("plane.urdf")
#Set friction for base link (-1) and all other links
p.changeDynamics(terrain_id, -1, lateralFriction=1.0)  # base link (e.g. new_start1)

for i in range(p.getNumJoints(terrain_id)):
    p.changeDynamics(terrain_id, i, lateralFriction=1.0)
'''
p.setPhysicsEngineParameter(
    numSolverIterations=150,
    contactBreakingThreshold=0.0001,
    jointFeedbackMode=1,  # Enable joint force feedback
    enableConeFriction=1,  # Better friction modeling
    enableFileCaching=1,
)


#...................TERRAIN LOADING....................#

if TASK == "RUBBLE":
    # Parameters
    N = 32                      # Heightfield resolution (N x N)
    block_size = 1  # Group size
    height_range = 0.1  # Max height difference

    grid_size = N // block_size + 1  # For corners
    corner_heights = np.random.uniform(0, height_range, (grid_size, grid_size))

    # Generate interpolated heightfield
    heightfield = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            bi = i // block_size
            bj = j // block_size
            lx = (i % block_size) / block_size
            ly = (j % block_size) / block_size

            h00 = corner_heights[bi, bj]
            h10 = corner_heights[bi+1, bj]
            h01 = corner_heights[bi, bj+1]
            h11 = corner_heights[bi+1, bj+1]

            h = (h00 * (1 - lx) * (1 - ly) +
                h10 * lx * (1 - ly) +
                h01 * (1 - lx) * ly +
                h11 * lx * ly)
            
            heightfield[i, j] = h

    # Flatten for PyBullet (row-major)
    heightfield_data = heightfield.flatten()

    # Create terrain shape
    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[0.2, 0.2, 1.0],
        heightfieldTextureScaling=(
            N - 1) / 2,
        heightfieldData=heightfield_data.tolist(),
        numHeightfieldRows=N,
        numHeightfieldColumns=N,
    )

    #textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")

    terrain1 = p.createMultiBody(0, terrain_shape)
    terrain2 = p.createMultiBody(0, terrain_shape, basePosition=[6, 0, 0])
    terrain3 = p.createMultiBody(0, terrain_shape, basePosition=[12, 0, -0.01])


    #p.changeVisualShape(terrain1, -1, textureUniqueId = textureId)
    #p.changeVisualShape(terrain2, -1, textureUniqueId = textureId)
    #p.changeVisualShape(terrain3, -1, textureUniqueId = textureId)


    p.changeDynamics(
        terrain1,
        -1,
        lateralFriction = 0.7
    )

    p.changeDynamics(
        terrain2,
        -1,
        lateralFriction = 0.7
    )

    p.changeDynamics(
        terrain3,
        -1,
        lateralFriction = 0.7
    )


    create_wall(0, -3, length=6.4)   
    create_wall(0, 3, length=6.4)

    create_wall(6.4, -3, length=6.4)   
    create_wall(6.4, 3, length=6.4)

    create_wall(12.8, -3, length=6.4)   
    create_wall(12.8, 3, length=6.4)

    create_wall(15, 0, length = 6.4, orientation = np.pi/2)
    create_wall(-3, 0, length = 6.4, orientation = np.pi/2)

    # --- START LINE text ---
    p.addUserDebugText(
        text="START LINE",
        textPosition=[start_threshold_x, 0,  0.5],
        textColorRGB=[0, 1, 0],
        textSize=1.5,
        lifeTime=0  # Permanent
    )

    start_line_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.01 / 2, 6 / 2, 0.01 / 2]
    )
    start_line_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.01 / 2, 6 / 2, 0.01 / 2],
        rgbaColor=[0, 1, 0, 1]  # Green
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=start_line_collision,
        baseVisualShapeIndex=start_line_visual,
        basePosition=[start_threshold_x, 0, 0.7]
    )

    # --- Floating 3D Text ---
    p.addUserDebugText(
        text="FINISH LINE",
        textPosition=[end_threshold_x, 0,  0.5],
        textColorRGB=[1, 0, 0],
        textSize=1.5,
        lifeTime=0  # 0 = permanent
    )


    end_line_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.01 / 2, 6 / 2, 0.01 / 2]
    )
    end_line_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.01 / 2, 6 / 2, 0.01 / 2],
        rgbaColor=[1, 0, 0, 1]  # Green
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=end_line_collision,
        baseVisualShapeIndex=end_line_visual,
        basePosition=[end_threshold_x, 0, 0.7]
    )

else:

    segment_length = 1
    segment_width = 3.0
    thickness = 0.05
    incline_angle_deg = 5
    incline_angle_rad = math.radians(incline_angle_deg)

    current_x = 0
    current_z = -0.1

    wall_height = 1.0
    wall_thickness = 0.05

    gray_color = [0.5, 0.5, 0.5, 1.0]  # RGBA: gray

    # Shared wall collision and visual shapes
    wall_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[segment_length, wall_thickness / 2, wall_height / 2]
    )
    wall_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[segment_length, wall_thickness / 2, wall_height / 2],
        rgbaColor=gray_color
    )

    while current_x < 20:
        # Incline segment
        height_diff = segment_length * math.tan(incline_angle_rad)

        center_x = current_x + (segment_length / 2) * math.cos(incline_angle_rad)
        center_z = current_z + (segment_length / 2) * math.sin(incline_angle_rad)

        incline_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[segment_length / 2, segment_width / 2, thickness / 2]
        )
        incline_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[segment_length / 2, segment_width / 2, thickness / 2],
            rgbaColor=gray_color
        )

        incline_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=incline_collision,
            baseVisualShapeIndex=incline_visual,
            basePosition=[center_x, 0, center_z],
            baseOrientation=p.getQuaternionFromEuler([0, -incline_angle_rad, 0])
        )
        p.changeDynamics(incline_body, -1, lateralFriction=1.0)

        # Add walls for incline
        for side in [-1, 1]:
            wall_y = side * (segment_width / 2 + wall_thickness / 2)
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=[center_x, wall_y, -0.1 + center_z + wall_height / 2],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )

        # Update x/z to end of incline
        current_x += segment_length * math.cos(incline_angle_rad)
        current_z += height_diff

        # Check if placing the flat segment would exceed x=16
        if current_x + segment_length / 2 > 16:
            break

        # Flat segment
        flat_center_x = current_x + segment_length / 2
        flat_center_z = current_z

        flat_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[segment_length / 2, segment_width / 2, thickness / 2]
        )
        flat_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[segment_length / 2, segment_width / 2, thickness / 2],
            rgbaColor=gray_color
        )

        flat_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=flat_collision,
            baseVisualShapeIndex=flat_visual,
            basePosition=[flat_center_x, 0, flat_center_z]
        )
        p.changeDynamics(flat_body, -1, lateralFriction=1.0)

        # Add walls for flat segment
        for side in [-1, 1]:
            wall_y = side * (segment_width / 2 + wall_thickness / 2)
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=[flat_center_x, wall_y, -0.1 + flat_center_z + wall_height / 2],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )

        # Move to next incline
        current_x += segment_length

    # --- Red line across the terrain at x = 15 ---
    finish_line_x = end_threshold_x
    line_thickness = 0.01
    line_height = 0.02

    line_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[line_thickness / 2, segment_width / 2, line_height / 2]
    )
    line_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[line_thickness / 2, segment_width / 2, line_height / 2],
        rgbaColor=[1, 0, 0, 1]  # Red
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=line_collision,
        baseVisualShapeIndex=line_visual,
        basePosition=[finish_line_x, 0, current_z + 0.6]
    )

    # --- Floating 3D Text ---
    p.addUserDebugText(
        text="FINISH LINE",
        textPosition=[finish_line_x, 0, current_z + 0.5],
        textColorRGB=[1, 0, 0],
        textSize=1.5,
        lifeTime=0  # 0 = permanent
    )


    # --- Green START line across the terrain at x = 2 ---
    start_line_x = start_threshold_x
    line_thickness = 0.01
    line_height = 0.02

    # Green line shape

    start_line_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[line_thickness / 2, 6 / 2, line_height / 2]
    )
    start_line_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[line_thickness / 2, 6 / 2, line_height / 2],
        rgbaColor=[0, 1, 0, 1]  # Green
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=start_line_collision,
        baseVisualShapeIndex=start_line_visual,
        basePosition=[start_line_x, 0, 0.5]
    )


    # --- START LINE text ---
    p.addUserDebugText(
        text="START LINE",
        textPosition=[start_line_x, 0, current_z + 0.3],
        textColorRGB=[0, 1, 0],
        textSize=1.5,
        lifeTime=0  # Permanent
    )



# ===== Robot Loading =====
snakebot = p.loadURDF(robot, [-2, 0, 0],)
start_pos = p.getBasePositionAndOrientation(snakebot)[0]
prev_pos = start_pos


# Get only REVOLUTE joints (ignore fixed joints)
joint_indices = [i for i in range(p.getNumJoints(snakebot)) 
                if p.getJointInfo(snakebot, i)[2] == p.JOINT_REVOLUTE]
num_joints = len(joint_indices)

num_links = num_joints + 1


for link_index in range(-1, num_joints):  # -1 is for the base link
    p.changeDynamics(
        snakebot,
        link_index,
        anisotropicFriction=frictionSet,  # [lateral, longitudinal, vertical]
        frictionAnchor=1, # Important for proper anisotropic behavior
        rollingFriction = 0.3 if GAIT=="INCHING" else 0,
        linearDamping = 0.1,
        angularDamping = 0.1,
        contactProcessingThreshold=0.001
    )


# ===== Motion Parameters =====

UNDULATION_AMPLITUDE = 0.6  # Max joint angle (radians)
UNDULATION_FREQUENCY = 1.2 # Wave frequency (Hz)
UNDULATION_PHASE_OFFSET = 2*math.pi/num_joints   # Phase difference between adjacent joints
STEER_BIAS = 0

SIDEWINDING_AMPLITUDE_YAW = 0.6
SIDEWINDING_AMPLITUDE_PITCH = 0.2
SIDEWINDING_SPATIAL_FREQUENCY = 2*math.pi/12
SIDEWINDING_TEMPORAL_FREQUENCY = 2.2*math.pi

INCHING_AMPLITUDE_YAW = 0
INCHING_AMPLITUDE_PITCH = 0.6
INCHING_SPATIAL_FREQUENCY = 2*math.pi/12
INCHING_TEMPORAL_FREQUENCY = 1*math.pi

ROLLING_AMPLITUDE_YAW = 0.1
ROLLING_AMPLITUDE_PITCH = 0.1
ROLLING_SPATIAL_FREQUENCY = 0.125*2*math.pi/12
ROLLING_TEMPORAL_FREQUENCY = 6
DIRECTION = -1

# ===== Torque/Control Settings =====
MAX_FORCE = 20  # Maximum motor torque (N.m)
MID_FORCE = 10  # Medium motor torque (N.m)
LOW_FORCE = 2

GAINS = {
    "position": 0.3,  # PID position gain
    "velocity": 0.8   # PID velocity gain
}



# Initialize motors in POSITION_CONTROL mode
for joint in joint_indices:
    p.setJointMotorControl2(
        snakebot,
        joint,
        p.POSITION_CONTROL,
        targetPosition=0,
        force=MAX_FORCE,
        positionGain=GAINS["position"],
        velocityGain=GAINS["velocity"]
    )



# ===== Main Control Loop =====
try:
    start_time = time.time()

    #CLEAN-SPAWN-PROCEDURE..........
    num_joints = p.getNumJoints(snakebot)
    p.changeDynamics(snakebot, -1, mass=0)  # base
    for j in range(num_joints):
        p.changeDynamics(snakebot, j, mass=0)


    tp_pos = [1.5, -0.5, 1]

    # Target orientation: for example, 15° yaw (rotation around z-axis)
    yaw_deg = 15
    yaw_rad = math.radians(yaw_deg)

    # Euler angles (roll, pitch, yaw)
    euler = [0, 0, rot]

    # Convert to quaternion
    target_orn = p.getQuaternionFromEuler(euler)

    # Teleport your robot
    p.resetBasePositionAndOrientation(snakebot, tp_pos, target_orn)

    num_joints = p.getNumJoints(snakebot)
    p.changeDynamics(snakebot, -1, mass=0.2)  # base
    for j in range(num_joints):
        p.changeDynamics(snakebot, j, mass=0.2)
    #..............................
    for _ in range(1200):  # Let physics settle
        p.stepSimulation() 

    ramp_duration = 5.0
    

    while True:
        current_time = time.time() - start_time

        keys = p.getKeyboardEvents()
        if ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
            follow_robot_camera(snakebot, head_link_index)
        

        head_link = p.getLinkState(snakebot, head_link_index)
        head_pos = head_link[0]
        head_x_pos = head_pos[0]

        check_start_logging(head_x_pos)  # Keep checking every frame


        if logging_enabled:
            # Auto stop after logging duration

            if head_x_pos > end_threshold_x: 
                print("🛑 Mission has been completed")
                break

            

            # --- Your logging routines ---
            force_sums = update_force_sums(snakebot, link_indices, force_sums,)
            
            step_count = update_steps(step_count)
            pos, _ = p.getBasePositionAndOrientation(snakebot)
            last_pos = pos  # keep updating the last known position

            for k in range(num_joints):
                joint_state = p.getJointState(snakebot, k)
                torque = joint_state[3]
                angular_velocity = joint_state[1]
                power = abs(torque * angular_velocity)
                total_energy += power * (1/240)

            current_pos = p.getBasePositionAndOrientation(snakebot)[0]
            step_dist = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
            total_dist += step_dist
            prev_pos = current_pos

            total_steps += 1
            contact_fraction_sum = update_contact_fraction(snakebot, contact_fraction_sum, num_links)


        if current_time < ramp_duration:
            current_force = (current_time / ramp_duration) * final_force
        else:
            current_force = final_force


    
    
        if GAIT == "UNDULATION":
            target_steer = 0
            keys = p.getKeyboardEvents()

            if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN:
                target_steer = 0.3
            elif ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN:
                target_steer = -0.3
            else:
                target_steer = 0

                
            # Smooth steering change
            STEER_BIAS += 0.05 * (target_steer - STEER_BIAS)


            # Clamp bias to reasonable range to avoid freakout
            STEER_BIAS = max(min(STEER_BIAS, 0.15), -0.15)
            JOINT_LIMIT = 1.5



            for i in range(len(joint_indices)):
                if True:
                    # Hirose wave formula with phase progression
                    phase = 2 * math.pi * UNDULATION_FREQUENCY * current_time + i * UNDULATION_PHASE_OFFSET
                    target_pos = UNDULATION_AMPLITUDE * math.sin(phase) + STEER_BIAS
                    target_pos = max(min(target_pos, JOINT_LIMIT), -JOINT_LIMIT)
                        
                    p.setJointMotorControl2(
                        snakebot,
                        i,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=current_force,
                        positionGain=GAINS["position"],
                        velocityGain=GAINS["velocity"]

                    )           
        elif GAIT == "SIDEWINDING":
            
            target_steer = 0
            direction = 0
            keys = p.getKeyboardEvents()

            if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN:
                target_steer = 0.6
                direction = 1
            elif ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN:
                target_steer = 0.6
                direction = -1
            else:
                target_steer = 0.03
                direction = 0

            # Smooth steering change
            STEER_BIAS += 0.5 * (target_steer - STEER_BIAS)
            JOINT_LIMIT = 1.5

            # Clamp bias to reasonable range to avoid freakout
            STEER_BIAS = max(min(STEER_BIAS, 0.3), -0.3)


                # Apply Travelling Waves in Perpendicular Planes
            for i in range(len(joint_indices)):

                if i%2==1:
                    # Hirose wave formula with phase progression
                    phase = SIDEWINDING_SPATIAL_FREQUENCY*i + SIDEWINDING_TEMPORAL_FREQUENCY*current_time
                    target_pos = SIDEWINDING_AMPLITUDE_YAW * math.sin(phase)
          
                    
                    if direction==0:
                        if i>6:
                            target_pos *= (1+ STEER_BIAS)

                    if direction<0:
                        if i<6:
                            target_pos *= (1+ STEER_BIAS) 
                            
                    if direction>0:
                        if TASK == "RUBBLE":
                            if i<6:
                                target_pos *= (1 - STEER_BIAS)
                            if i>6:
                                target_pos *= (1+ STEER_BIAS)
                        elif TASK == "HIGH-FRICTION RAMP-STAIRS":
                            if i>6:
                                target_pos *= (1+ STEER_BIAS)
                


                    p.setJointMotorControl2(
                        snakebot,
                        i,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=current_force,
                        positionGain=GAINS["position"],
                        velocityGain=GAINS["velocity"]

                    )

                elif i%2==0:
                    # Hirose wave formula with phase progression
                    phase = SIDEWINDING_SPATIAL_FREQUENCY*i + SIDEWINDING_TEMPORAL_FREQUENCY*current_time 
                    target_pos = SIDEWINDING_AMPLITUDE_PITCH * math.cos(phase)

                    
                    
                    p.setJointMotorControl2(
                        snakebot,
                        i,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=current_force,
                        positionGain=GAINS["position"],
                        velocityGain=GAINS["velocity"]

                    )

        elif GAIT == "INCHING":

            for i in range(0,11,1): 

                #Yaw Joint Progression (Minor)
                if i%2==1:
                    
                    phase = -INCHING_SPATIAL_FREQUENCY*i + INCHING_TEMPORAL_FREQUENCY*current_time
                    target_pos = INCHING_AMPLITUDE_YAW * math.sin(phase)
                    
                    if i==0:
                        phase = phase/2
                    
                    p.setJointMotorControl2(
                        snakebot,
                        i,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=LOW_FORCE,
                        positionGain=GAINS["position"],
                        velocityGain=GAINS["velocity"]

                    )

                # Pitch Joint Progression (Major)
                if i%2==0:
                    
                    phase = -INCHING_SPATIAL_FREQUENCY*(i) - INCHING_TEMPORAL_FREQUENCY*current_time + math.pi/2
                    target_pos = INCHING_AMPLITUDE_PITCH * math.sin(phase) 

                    if i==0:
                        target_pos = target_pos+ (math.pi/12)
                    elif i==2:
                        target_pos = target_pos+ (math.pi/12)
                    elif i ==4:
                        target_pos = target_pos+ (math.pi/12)
                    elif i==6:
                        target_pos = (math.pi/3)*target_pos
                    elif i==8:
                        target_pos= (math.pi/3)*target_pos
                    elif i==10:
                        target_pos = (math.pi/3)*target_pos
                    
                    p.setJointMotorControl2(
                        snakebot,
                        i,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=LOW_FORCE,
                        positionGain=GAINS["position"],
                        velocityGain=GAINS["velocity"]

                    )


        p.stepSimulation()       
        time.sleep(1./240.)  # Sync with real-time (240 Hz)


        
except KeyboardInterrupt:
    pass

finally:
    if logging_enabled and step_count > 0:

        print("\n\n#=== REPORT ===\n\n")
        #duty_factors = {link: duration / step_count for link, duration in contact_durations.items()}
        #average_duty_factor = sum(duty_factors.values()) / len(duty_factors)
        #print("Average duty factor:", average_duty_factor)


        
        average_csi = contact_fraction_sum / step_count
        print(f"\nAverage CSI: {average_csi:.3f}")

        displacement_before_failure = head_x_pos - start_threshold_x
        print(f"\nNet Displacement: (Before Failure):  {displacement_before_failure}")

        cost_of_transport = total_energy / (total_dist * (13*0.15*9.81))
        print(f"\nCost of Transport: {cost_of_transport:.3f} ")

        time_of_completion = step_count/240 
        print(f"\nTime of Completion: {time_of_completion:.3f} ")
        
        net_displacement = last_pos[0] - start_pos[0]  # forward progress
        effective_velocity = net_displacement / (step_count/240)
        print(f"\nEffective Velocity: {effective_velocity:.3f} m/s ({effective_velocity * 3.6:.2f} km/h)\n")

        avg_forces = {link: force_sums[link]/step_count for link in force_sums}
        #print("Final average forces:", avg_forces)

        forces_values = list(avg_forces.values())
        sd = np.std(forces_values)
        mean = np.mean(forces_values)
        print(f"Standard Deviation of average forces: {sd:.3f}")
        print(f"Mean of average forces: {mean:.3f}")

        cv = sd/mean
        print(f"Coefficient of variation for force distribution {cv:.3f}")

        

        print("\n\n==============")


        min_force = min(avg_forces.values())
        max_force = max(avg_forces.values())
        norm = plt.Normalize(vmin=min_force, vmax=max_force)
        cmap = plt.cm.hot # you can choose any matplotlib colormap

        fig, ax = plt.subplots(figsize=(12, 3))

        # Cylinder “geometry” parameters
        cyl_length = 0.8
        cyl_height = 0.4
        spacing = 1.0  # distance between the centers of adjacent cylinders

        min_force = min(avg_forces.values())
        max_force = max(avg_forces.values())
        norm = plt.Normalize(vmin=min_force, vmax=max_force)

        for i, (link, force) in enumerate(avg_forces.items()):
            x_center = i * spacing
            y_center = 0

            color = cmap(norm(force))

            rect = Rectangle(
                (x_center - cyl_length / 2, y_center - cyl_height / 2),
                cyl_length,
                cyl_height,
                linewidth=1,
                edgecolor='black',
                facecolor=color
            )
            ax.add_patch(rect)

            left_circle = Circle(
                (x_center - cyl_length / 2, y_center),
                cyl_height / 2,
                edgecolor='black',
                facecolor=color
            )
            ax.add_patch(left_circle)

            right_circle = Circle(
                (x_center + cyl_length / 2, y_center),
                cyl_height / 2,
                edgecolor='black',
                facecolor=color
            )
            ax.add_patch(right_circle)

            ax.text(x_center, y_center - 0.6, f"{link}", ha='center', va='center')


        # Adjust view limits so all cylinders are visible
        ax.set_xlim(-spacing, spacing * len(avg_forces))
        ax.set_ylim(-1, 1)
        ax.axis('off')

        # Add a horizontal colorbar to show force→color mapping
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label('Average Force (N)')

        plt.show()


        trial_data = {
        'trial': trial_num,
        'task': TASK,
        'gait': GAIT,
        'Force CV': cv,
        'Cost of Transport': cost_of_transport,
        'Average CSI': average_csi,
        #'Distance Traveled': displacement_before_failure,
        #'Success/Fail': 0,
        #'Fail Reason': "N/A"
        'Time of Completion': time_of_completion
        
        }

        # Save/append to CSV
        file_path = "snakebot_results.csv"
        df = pd.DataFrame([trial_data])  # Note: [trial_data] is a list of one row

        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)
        


    p.disconnect()