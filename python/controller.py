"""
Nonlinear Controller
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0


class NonlinearController(object):
    def __init__(
            self,
            Kp_gain_pos=6,  # 12.0, 6
            Kp_gain_vel=4.8,  # 8.0, 4
            Kp_gain_alt=4.0,
            Kp_gain_i=2.0,

            Kp_gain_roll=5.7,  # 6.5, 8
            Kp_gain_pitch=6,  # 6.5, 8
            Kp_gain_yaw=3.5,  # 4.5

            Kp_p=17.8,  # 10, 20
            Kp_q=20,  # 10, 20
            Kp_r=13,


            max_ascending_rate=5,
            max_descending_rate=2,
            max_speed=5.0
    ):

        self.Kp_gain_pos = Kp_gain_pos
        self.Kp_gain_vel = Kp_gain_vel
        self.Kp_gain_alt = Kp_gain_alt
        self.Kp_gain_i = Kp_gain_i
        self.Kp_gain_roll = Kp_gain_roll
        self.Kp_gain_pitch = Kp_gain_pitch
        self.Kp_gain_yaw = Kp_gain_yaw
        self.Kp_r = Kp_r
        self.Kp_p = Kp_p
        self.Kp_q = Kp_q
        self.max_ascending_rate = max_ascending_rate
        self.max_descending_rate = max_descending_rate
        self.max_speed = max_speed



    def trajectory_control(self, position_trajectory,
                           yaw_trajectory,
                           time_trajectory,
                           current_time):
        """Generate a commanded position, velocity and yaw based on the
            trajectory

        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the
                            position and yaw commands
            current_time: float corresponding to the current time in seconds

        Returns: tuple (commanded position, commanded velocity, commanded yaw)

        """
        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]

        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]

            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]

        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]

                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        position_cmd = (position1 - position0) * \
                       (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)

        return (position_cmd, velocity_cmd, yaw_cmd)

    def lateral_position_control(self, local_position_cmd,
                         local_velocity_cmd, local_position, local_velocity,
                         acceleration_ff=np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the
           local frame
        Args:
            local_position_cmd: desired 2D position in local frame
                                [north, east]
            local_velocity_cmd: desired 2D velocity in local frame
                                [north_velocity, east_velocity]
            local_position: vehicle position in the local frame
                            [north, east]
            local_velocity: vehicle velocity in the local frame
                            [north_velocity, east_velocity]
            acceleration_ff: feedforward acceleration command

        Returns: desired vehicle 2D acceleration in the local frame
                [north, east]
        """
        pos_err = local_position_cmd - local_position
        vel_error = local_velocity_cmd - local_velocity
        velocity_cmd = self.Kp_gain_pos * pos_err

        # Find Euclidean denominator Norm for Veleocity
        velocity_norm = np.sqrt(velocity_cmd[0] * velocity_cmd[0] + velocity_cmd[1] * velocity_cmd[1])

        if velocity_norm > self.max_speed:
            velocity_cmd = velocity_cmd * self.max_speed / velocity_norm

        acceleration_cmd = acceleration_ff + self.Kp_gain_pos * pos_err + self.Kp_gain_vel * vel_error

        return acceleration_cmd

    def altitude_control(self, altitude_cmd,
                         vertical_velocity_cmd,
                         altitude,
                         vertical_velocity,
                         attitude,
                         acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command
        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """

        # Get rotation matrix  from  attitude
        rot_mat = euler2RM(attitude[0], attitude[1], attitude[2])
        b_z = rot_mat[2, 2]

        altitude_err = altitude_cmd - altitude

        integral_err = self.Kp_gain_alt * altitude_err + vertical_velocity_cmd

        # Margin the  ascending and descending rate
        if integral_err > self.max_ascending_rate:
            integral_err = self.max_ascending_rate
        elif integral_err < -self.max_descending_rate:
            integral_err = -self.max_descending_rate

        acceleration_cmd = DRONE_MASS_KG * acceleration_ff + self.Kp_gain_i * (integral_err - vertical_velocity)

        thrust = acceleration_cmd / b_z

        if thrust > MAX_THRUST:
            thrust = MAX_THRUST
        elif thrust < 0.0:
            thrust = 0.0
        return thrust

    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            target_acceleration: 2-element numpy array
                        (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll,pitch,yaw) in radians
            thrust_cmd: vehicle thruts command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and
                pitchrate (q) commands in radians/s
        """

        R = euler2RM(attitude[0], attitude[1], attitude[2])
        c_d = thrust_cmd / DRONE_MASS_KG

        if thrust_cmd > 0.0:
            accel_p_term = -min(max(acceleration_cmd[0].item() / c_d, -1.0), 1.0)
            accel_q_term = -min(max(acceleration_cmd[1].item() / c_d, -1.0), 1.0)

            p_cmd = (1 / R[2, 2]) * (-R[1, 0] * self.Kp_gain_roll * (R[0, 2] - accel_p_term) +  R[0, 0] * self.Kp_gain_pitch * (R[1, 2] - accel_q_term))


            q_cmd = (1 / R[2, 2]) *  (-R[1, 1] * self.Kp_gain_pitch * (R[0, 2] - accel_p_term) +  R[0, 1] * self.Kp_gain_pitch * (R[1, 2] - accel_q_term))


        else:  # Otherwise command no rate
            p_cmd = 0.0
            q_cmd = 0.0
            thrust_cmd = 0.0
        return np.array([p_cmd, q_cmd])



    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame

        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd)
                        in radians/second^2
            attitude: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and
                yaw moment commands in Newtons*meters
        """
        Kp_body_rate = np.array([self.Kp_p, self.Kp_q, self.Kp_r])
        body_rate_error = body_rate_cmd - body_rate

        moment_cmd = MOI * np.multiply(Kp_body_rate, body_rate_error)

        if np.linalg.norm(moment_cmd) > MAX_TORQUE:
            moment_cmd = moment_cmd * MAX_TORQUE / np.linalg.norm(moment_cmd)
        return moment_cmd

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """

        # Margin the angle within  0 to 2*pi
        yaw_cmd = np.mod(yaw_cmd, 2.0 * np.pi)

        yaw_error = yaw_cmd - yaw

        if yaw_error > np.pi:
            yaw_error = yaw_error - 2.0 * np.pi
        elif yaw_error < -np.pi:
            yaw_error = yaw_error + 2.0 * np.pi

        yawrate_cmd = self.Kp_gain_yaw * yaw_error
        return yawrate_cmd