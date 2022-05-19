"""
Move the jao2 Mujoco arm to a target position.
The simulation ends after 1500 time steps, and the
trajectory of the end-effector is plotted in 3D.
"""
import sys
import traceback

import glfw
import numpy as np

from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.controllers import OSC, Damping
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils import transformations


file = "../../../abr_control/arms/ur5dual/environments/google_dual_ur.xml"
robot_config = MujocoConfig(xml_file=file,folder='.')

# create our Mujoco interface
interface = Mujoco(robot_config, dt=0.001)
joint_names = [["joint0","joint1","joint2","joint3","joint4","joint5"],
               ["joint0_right","joint1_right","joint2_right","joint3_right","joint4_right","joint5_right"]]
interface.connect(joint_names)

# damp the movements of the arm
damping = Damping(robot_config, kv=10)
# instantiate controller for both arms
ctrl_left = OSC(
    robot_config,
    kp=200,
    null_controllers=[damping],
    vmax=[0.5, 0],  # [m/s, rad/s]
    # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[True, True, True, False, False, False],
    arm_num = 0
)

ctrl_right = OSC(
    robot_config,
    kp=200,
    null_controllers=[damping],
    vmax=[0.5, 0],  # [m/s, rad/s]
    # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[True, True, True, False, False, False],
    arm_num = 1
)

# set up lists for tracking data
ee_track_left = []
ee_track_right = []

target_geom_id_left = interface.sim.model.geom_name2id("target")
target_geom_id_right = interface.sim.model.geom_name2id("target1")

green = [0, 0.9, 0, 0.5]
red = [0.9, 0, 0, 0.5]


def gen_target_left(interface):
    target_xyz_left = np.array(
        [
            np.random.random() * 0.4 + 0.1,
            np.random.random() * 0.5 + 0.2,
            np.random.random() * 0.4 + 0.15,
        ]
    )
    interface.set_mocap_xyz(name="target", xyz=target_xyz_left)
    
def gen_target_right(interface):
    target_xyz_right = np.array(
        [
            np.random.random() * -0.4 - 0.1,
            np.random.random() * 0.5 + 0.2,
            np.random.random() * 0.4 + 0.15,
        ]
    )
    interface.set_mocap_xyz(name="target1", xyz=target_xyz_right)

try:

    # make the target offset from that start position
    gen_target_left(interface)
    gen_target_right(interface)

    count = 0.0
    print("\nSimulation starting...\n")
    while 1:
        # get joint angle and velocity feedback
        feedback_left = interface.get_feedback(arm_num=0)
        feedback_right = interface.get_feedback(arm_num=1)

        target_left = np.hstack(
            [
                interface.get_xyz("target"),
                transformations.euler_from_quaternion(
                    interface.get_orientation("target"), "rxyz"
                ),
            ]
        )
        
        target_right = np.hstack(
            [
                interface.get_xyz("target1"),
                transformations.euler_from_quaternion(
                    interface.get_orientation("target1"), "rxyz"
                ),
            ]
        )

        # calculate the control signal
        u_left = ctrl_left.generate(
            q=feedback_left["q"],
            dq=feedback_left["dq"],
            target=target_left,
        )
        
        u_right = ctrl_right.generate(
            q=feedback_right["q"],
            dq=feedback_right["dq"],
            target=target_right,
        )

        # send forces into Mujoco, step the sim forward
        interface.send_forces(u_left,arm_num=0)
        interface.send_forces(u_right,arm_num=1)

        # calculate end-effector position
        ee_xyz_left = robot_config.Tx("EE", q=feedback_left["q"])
        ee_xyz_right = robot_config.Tx("EE_right",q=feedback_right["q"])
        
        # track error
        error_left = np.linalg.norm(ee_xyz_left - target_left[:3])
        error_right = np.linalg.norm(ee_xyz_right - target_right[:3])
        
            
        if error_left < 0.02:
            gen_target_left(interface)

        if error_right < 0.02:
            gen_target_right(interface)
            

except:
    print(traceback.format_exc())
