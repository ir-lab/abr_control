"""
Running the joint controller with an inverse kinematics path planner
for a Mujoco simulation. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which changes every n time steps.
"""
import glfw
import numpy as np

from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.controllers import path_planners
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils import transformations

import mujoco_py as mjp

import csv

# initialize our robot config for the Dual arm
file = "../../../abr_control/arms/ur5dual/environments/google_dual_ur.xml"
robot_config = MujocoConfig(xml_file=file,folder='.',use_sim_state=False)

# create our interface
dt = 0.001
interface = Mujoco(robot_config, dt=dt)
joint_names = [["joint0_mra_id_1","joint1_mra_id_1","joint2_mra_id_1","joint3_mra_id_1","joint4_mra_id_1","joint5_mra_id_1"],
                ["joint0_mra_id_2","joint1_mra_id_2","joint2_mra_id_2","joint3_mra_id_2","joint4_mra_id_2","joint5_mra_id_2"]]
arm_0 = 0
arm_1 = 1 
interface.connect(joint_names)

# create our path planner for each aarm
n_timesteps = 1000
path_planner_0 = path_planners.InverseKinematics(robot_config,arm_num=arm_0)
path_planner_1 = path_planners.InverseKinematics(robot_config,arm_num=arm_1)

try:
    print("\nSimulation starting...")
    print("Click to move the target.\n")
    count = 0
    while 1:

        if count % n_timesteps == 0:
            feedback_1 = interface.get_feedback(arm_num=arm_1)
            target_xyz_1 = np.array(
                [
                    np.random.random() * 0.5 - 0.2,
                    np.random.random() * 0.5 - 0.5,
                    np.random.random() * 0.5 + 0.25,
                ]
            )
            R_1 = robot_config.R("EE_mra_id_2", q=feedback_1["q"])
            target_orientation_1 = transformations.euler_from_matrix(R_1, "sxyz")
            
            feedback_0 = interface.get_feedback(arm_num=arm_0)
            target_xyz_0 = np.array(
                [
                    np.random.random() * 0.5 + 0.2,
                    np.random.random() * 0.5 + 0.2,
                    np.random.random() * 0.5 + 0.25,
                ]
            )
            R_0 = robot_config.R("EE_mra_id_1", q=feedback_0["q"])
            target_orientation_0 = transformations.euler_from_matrix(R_0, "sxyz")
            
            # update the position of the target
            interface.set_mocap_xyz("target", target_xyz_0)
            interface.set_mocap_xyz("target1",target_xyz_1)

            # can use 3 different methods to calculate inverse kinematics
            # see inverse_kinematics.py file for details
            path_planner_0.generate_path(
                position=feedback_0["q"],
                target_position=np.hstack([target_xyz_0, target_orientation_0]),
                method=3,
                dt=0.005,
                n_timesteps=n_timesteps,
                plot=False,
            )
            
            path_planner_1.generate_path(
                position=feedback_1["q"],
                target_position=np.hstack([target_xyz_1, target_orientation_1]),
                method=3,
                dt=0.005,
                n_timesteps=n_timesteps,
                plot=False,
            )
        
        target_0 = path_planner_0.next()[0]
        target_1 = path_planner_1.next()[0]

        
        # use position control
        interface.send_target_angles(target_0,arm_0)
        interface.send_target_angles(target_1,arm_1)

        interface.viewer.render()
        count += 1

finally:
    # stop and reset the simulation
    interface.disconnect()

    print("Simulation terminated...")
