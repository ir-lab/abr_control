#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:49:56 2022

@author: raviswarooprayavarapu
"""

"""
A basic script for connecting to the arm and putting it in floating
mode, which only compensates for gravity. The end-effector position
is recorded and plotted when the script is exited (with ctrl-c).

In this example, the floating controller is applied in the joint space
"""
import sys
import traceback

import glfw
import numpy as np

from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.controllers import Floating
from abr_control.interfaces.mujoco import Mujoco

file = "Google Dual UR/environments/google_dual_ur.xml"
# initialize our robot config
robot_config = arm(xml_file=file,folder='.')

# create the Mujoco interface and connect up
interface = Mujoco(robot_config, dt=0.001)


joint_names = [["joint0","joint1","joint2","joint3","joint4","joint5"],
               ["joint0_right","joint1_right","joint2_right","joint3_right","joint4_right","joint5_right"]]

interface.connect(joint_names)
interface.send_target_angles(robot_config.START_ANGLES)

# instantiate the controller
ctrlr_0 = Floating(robot_config, task_space=False, dynamic=True,arm_num=0)
ctrlr_1 = Floating(robot_config, task_space=False, dynamic=True,arm_num=1)

# set up arrays for tracking end-effector and target position
ee_track = []
q_track = []


try:


    print("\nSimulation starting...\n")

    while 1:
        # get joint angle and velocity feedback
        feedback_0 = interface.get_feedback(arm_num = 0)
        feedback_1 = interface.get_feedback(arm_num = 1)

        # calculate the control signal
        u_0 = ctrlr_0.generate(q=feedback_0["q"], dq=feedback_0["dq"])
        u_1 = ctrlr_1.generate(q=feedback_1["q"], dq=feedback_1["dq"])
        
        print("u_0 ", u_0)
        print("u_1 ", u_1 )

        # send forces into Mujoco
        interface.send_forces(u_0,arm_num=0)
        interface.send_forces(u_1,arm_num=1)

        # calculate the position of the hand
        hand_xyz_0 = robot_config.Tx("EE", q=feedback_0["q"])
        hand_xyz_1 = robot_config.Tx("EE_right", q=feedback_1["q"])
        
        # track end effector position
        ee_track.append(np.copy(hand_xyz_0))
        q_track.append(np.copy(feedback_0["q"]))

except:
    print(traceback.format_exc())

finally:
    # close the connection to the arm
    interface.disconnect()

    print("Simulation terminated...")

    ee_track = np.array(ee_track)

    if ee_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611

        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(211)
        ax1.set_title("Joint Angles")
        ax1.set_ylabel("Angle (rad)")
        ax1.set_xlabel("Time (ms)")
        ax1.plot(q_track)
        ax1.legend()

        ax2 = fig.add_subplot(212, projection="3d")
        ax2.set_title("End-Effector Trajectory")
        ax2.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label="ee_xyz")
        ax2.legend()
        plt.show()