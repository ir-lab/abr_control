import numpy as np

from .controller import Controller


class Damping(Controller):
    """Base class for common null space controllers.

    Parameters
    ----------
    robot_config : class instance
        contains all relevant information about the arm
        such as: number of joints, number of links, mass information etc.
    """

    def __init__(self, robot_config, kv,arm_num=0):
        super().__init__(robot_config)

        self.kv = kv
        self.arm_num = arm_num

    def generate(self, q, dq):
        """Generates the control signal

        q : np.array
          the current joint angles [radians]
        dq : np.array
          the current joint angle velocity [radians/second]
        """

        # calculate joint space inertia matrix
        M = self.robot_config.M(q=q,arm_num=self.arm_num)
        return np.dot(M, -self.kv * dq)
