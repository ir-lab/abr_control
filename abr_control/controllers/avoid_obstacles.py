import numpy as np

from .controller import Controller


class AvoidObstacles(Controller):
    """Implements an obstacle avoidance algorithm from (Khatib, 1987).

    Parameters
    ----------
    robot_config: class instance
        contains all relevant information about the arm
        such as number of joints, number of links, mass information etc.
    obstacles: list of list of floats
        a list of obstacles, where the obstacles are a list of the
        corresponding Cartesian coordinates and obstacle radius [metres]
        ex: obstacles = [obs1, obs2, obs3] where obs1 = [x1, y1, z1, radius]
    threshold: float, optional (Default: 0.2)
        how close is the system allowed to get to obstacles
    gain: scalar, optional (Default: 1)
        a scaling term on the strength of the avoidance signal
    maximum: scalar, optional (Default: 100)
        a maximum value at which to clip the output
    """

    def __init__(
        self, robot_config, obstacles=None, threshold=0.2, gain=1, maximum=500, arm_num=0
    ):
        super().__init__(robot_config)

        self.threshold = threshold
        self.gain = gain
        self.maximum = maximum
        self.arm_num = arm_num
        self.body_name = robot_config.arm[arm_num].ee

        obstacles = [] if obstacles is None else obstacles
        self.obstacles = np.array(obstacles)

    def generate(self, q, dq=None):
        """Generates the control signal

        Parameters
        ----------
        q: np.array
            the current joint angles [radians]
        dq: np.array
          the current joint angle velocity [radians/second]
          NOTE: Not used in this function
        """

        u_psp = np.zeros(self.robot_config.N_JOINTS[self.arm_num])

        # calculate joint space inertia matrix
        M = self.robot_config.M(q=q,arm_num=self.arm_num)

        # add in obstacle avoidance
        for obstacle in self.obstacles:
            # our vertex of interest is the center point of the obstacle
            v = np.array(obstacle[:3])

            # find the closest point of each link to the obstacle
            for ii in range(self.robot_config.N_JOINTS[self.arm_num]):
                # get the start and end-points of the arm segment
                p1 = self.robot_config.Tx(f"joint{ii}", q=q)
                if ii == self.robot_config.N_JOINTS[self.arm_num] - 1:
                    p2 = self.robot_config.Tx(self.body_name, q=q)
                else:
                    p2 = self.robot_config.Tx(f"joint{ii+1}", q=q)

                # calculate minimum distance from arm segment to obstacle
                # the vector of our line
                vec_line = p2 - p1
                # the vector from the obstacle to the first line point
                vec_ob_line = v - p1
                # calculate the projection normalized by length of arm segment
                projection = np.dot(vec_ob_line, vec_line) / np.sum((vec_line) ** 2)
                if projection < 0:
                    # then closest point is the start of the segment
                    closest = p1
                elif projection > 1:
                    # then closest point is the end of the segment
                    closest = p2
                else:
                    closest = p1 + projection * vec_line
                # calculate distance from obstacle vertex to the closest point
                dist = np.sqrt(np.sum((v - closest) ** 2))
                # account for size of obstacle
                # also set a minimum distance so the control signal
                # doesn't grow unbounded, value chosen empirically
                rho = max(dist - obstacle[3], self.threshold / 50)

                if rho < self.threshold:

                    eta = 0.02  # feel like i saw 4 somewhere in the paper
                    drhodx = (v - closest) / rho
                    Fpsp = (
                        eta
                        * (1.0 / rho - 1.0 / self.threshold)
                        * 1.0
                        / rho ** 1.5
                        * drhodx
                    )

                    # get offset of closest point from link's reference frame
                    # NOTE: the relevant link is i+1, because the configuration
                    # scripts are set up so link 0 is from origin to joint 0
                    T_inv = self.robot_config.T_inv(f"link{ii+1}", q=q)
                    m = np.dot(T_inv, np.hstack([closest, [1]]))[:-1]
                    # calculate the Jacobian for this point
                    Jpsp = self.robot_config.J(f"link{ii+1}", x=m, q=q, arm_num=self.arm_num)[:3]

                    # calculate the inertia matrix for the
                    # point subjected to the potential space
                    Mxpsp_inv = np.dot(Jpsp, np.dot(np.linalg.inv(M), Jpsp.T))
                    # using the rcond to set singular values < thresh to 0
                    # is slightly faster than doing it manually with svd
                    Mxpsp = np.linalg.pinv(Mxpsp_inv, rcond=0.01)

                    u_psp += -1 * np.dot(Jpsp.T, np.dot(Mxpsp, Fpsp))

        return np.clip(u_psp * self.gain, -self.maximum, self.maximum)

    def set_obstacles(self, obstacles):
        """Specify the locations of the obstacles to avoid

        Parameters
        ----------
        obstacles: list of list of floats
            a list of obstacles, where the obstacles are a list of the
            corresponding Cartesian coordinates and obstacle radius [metres]
            ex: obstacles = [obs1, obs2, obs3] where obs1 = [x1, y1, z1, radius]
        """

        self.obstacles = np.copy(obstacles)
