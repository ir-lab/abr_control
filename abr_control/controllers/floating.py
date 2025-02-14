import numpy as np

from .controller import Controller


class Floating(Controller):
    """Implements a controller to compensate for gravity
    Only compensates for the effects of gravity on the arm. The arm will
    remain compliant and hold whatever position it is placed in (as long
    as an accurate mass / inertia model is provided)
    Parameters
    ----------
    robot_config: class instance
        contains all relevant information about the arm
        such as: number of joints, number of links, mass information etc.
    task_space: boolean, optional (Default: False)
        if True, perform the calculation to cancel out the effects of
        gravity in task-space
    """

    def __init__(self, robot_config, dynamic=False, task_space=False, arm_num=0):
        super().__init__(robot_config)

        self.dynamic = dynamic
        self.task_space = task_space
        self.arm_num = arm_num
        self.body_name = robot_config.arm[arm_num].ee

    def generate(self, q, dq=None):
        """Generates the control signal to compensate for gravity
        Parameters
        ----------
        q : float numpy.array
            the current joint angles [radians]
        dq : float numpy.array
            the current joint velocities [radians/second]
        """
        # calculate the effect of gravity in joint space
        g = self.robot_config.g(q,arm_num=self.arm_num)

        if self.task_space:
            # get the Jacobian
            J = self.robot_config.J(self.body_name, q)[:3]

            # calculate the inertia matrix in joint space
            M = self.robot_config.M(q,self.arm_num)
            # calculate the inertia matrix in task space
            M_inv = np.linalg.inv(M)

            Mx_inv = np.dot(J, np.dot(M_inv, J.T))
            if abs(np.linalg.det(Mx_inv)) > 1e-3:
                Mx = np.linalg.inv(Mx_inv)
            else:
                # using the rcond to set singular values < thresh to 0
                # is slightly faster than doing it manually with svd
                # singular values < (rcond * max(singular_values)) set to 0
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-4)

            # calculate the effect of gravity in joint space
            Jbar = np.dot(M_inv, np.dot(J.T, Mx))
            u_task = -1 * np.dot(Jbar.T, g)
            u = np.dot(J.T, u_task)
        else:
            # generate the control signal in joint space
            u = -g
            M = None

        if self.dynamic:
            # compensate for current velocity
            M = self.robot_config.M(q,arm_num=self.arm_num) if M is None else M
            u -= np.dot(M, dq)

        return u
