#!/usr/bin/env python3
"""Reference low-level passive FTC around policy.onnx, numpy + onnxruntime only (no JAX, no crazyflow).

This is the file a ROS2 node ports. It is a numpy copy of ftc.high_level (uav_ftc PID position controller +
primary-axis attitude controller), ftc.observe (27-d observation, training scaling) and the action mapping of
ftc.step, in the same order, so it reproduces the training-time controller exactly. See policies/README.md.

    pftc = PFTC("policies/x500v2_i012/policy.onnx")
    thrust = pftc.step(p, v, q_xyzw, w, acc_b, target)   # call at 100 Hz; N per rotor, PX4 motor order
    omega = PFTC.rotor_speed(thrust)                      # rad/s for Gazebo's MulticopterMotorModel

Inputs (world frame z up, body frame x forward / y left / z up): p, v [m, m/s] world; q_xyzw body-to-world;
w [rad/s] body rates; acc_b [m/s^2] specific force in the body frame, what an IMU reads (+9.81 on z at rest).
"""
import numpy as np
import onnxruntime

DT = 0.01  # s, policy and high-level rate (100 Hz)
MASS, G = 1.40, 9.81  # x500v2
F_MAX = 11.8  # N per rotor
F_HOVER = MASS * G / 4
MOTOR_CONSTANT = 1.18e-5  # N/(rad/s)^2, Gazebo custom_x500 motorConstant
F_FLOOR = 0.27  # N, PX4 armed-motor floor (150 rad/s); the policies from i017 on are trained with it


def lpf(prev, x, tau):
    return (DT * x + tau * prev) / (tau + DT)


def rotmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class PFTC:
    def __init__(self, onnx_path, floor=F_FLOOR):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.floor = floor  # minimum rotor command, N (0 reproduces the nominal training sim)
        self.reset()

    def reset(self):
        """State of a drone hovering at rest (same as ftc.reset_where)."""
        self.u = np.full(4, F_HOVER)
        self.u_lpf = np.full(4, F_HOVER)  # low-passed command, tau 0.025 s
        self.w_prev = np.zeros(3)
        self.dw_lpf = np.zeros(3)  # filtered angular acceleration, tau 0.12 s
        self.n_last = np.array([0.0, 0.0, 1.0])  # last desired primary axis
        self.hl_lpf = np.array([0.0, 0.0, G])  # low-passed [p_des, q_des, a_z_des]
        self.first = True

    def high_level(self, p, v, R, wz, target):
        """uav_ftc PositionController + PrimaryAxisAttitudeController, as ftc.high_level."""
        vel_t = np.clip(target - p, -10.0, 10.0)  # kp_pos = 1
        acc = np.array([2.0, 2.0, 6.0]) * (vel_t - v)  # kp_vel
        max_lat = G * np.tan(10.0 / 57.3)  # 10 deg max tilt command
        acc[:2] /= max(np.linalg.norm(acc[:2]) / max_lat, 1.0)
        acc[2] = np.clip(acc[2], -5.0, 5.0)
        thrust_dir = acc + np.array([0.0, 0.0, G])
        n_des = thrust_dir / np.linalg.norm(thrust_dir)
        n_dot = np.clip((n_des - self.n_last) / DT, -0.5, 0.5)
        h = R.T @ n_des  # desired primary axis in the body frame
        h3 = max(h[2], 0.1)  # guard: the original divides by h3, which blows up when tumbling
        nd = (R.T @ n_dot)[:2]
        tmp = -5.0 * h[:2] - wz * np.array([h[1], -h[0]]) - nd  # kx = ky = 5
        cmd = np.array([tmp[1] / h3, -tmp[0] / h3, np.linalg.norm(thrust_dir)])
        self.n_last, self.hl_lpf = n_des, lpf(self.hl_lpf, cmd, 1.0 / 50)  # BW = 50

    def observation(self, p, v, q_xyzw, w, acc_b, target):
        q = np.asarray(q_xyzw, float) * np.sign(q_xyzw[3] + 1e-9)  # q and -q are the same attitude
        hl = np.concatenate([np.clip(self.hl_lpf[:2], -10, 10), np.clip(self.hl_lpf[2:], -50, 50)])
        return np.concatenate([
            (p - target) / 5, v / 5, q[[3, 0, 1, 2]], w / np.array([5.0, 5.0, 20.0]),
            self.u_lpf / 10, acc_b / np.array([5.0, 5.0, 20.0]),
            hl[:2] / 5, [0.0], hl[2:] / 20, self.dw_lpf / 50]).astype(np.float32)

    def _update(self, p, v, q_xyzw, w, target):
        """Filters and high-level controller; ftc.step updates these after each physics step."""
        if not self.first:
            self.dw_lpf = (w - self.w_prev + 0.12 * self.dw_lpf) / (0.12 + DT)
            self.high_level(p, v, rotmat(q_xyzw), w[2], target)
        self.first, self.w_prev = False, w

    def action(self, p, v, q_xyzw, w, acc_b, target):
        """Policy output in [-1, 1] (0 = hover thrust). Updates the controller state; call once per 10 ms."""
        p, v, w, acc_b, target = (np.asarray(x, float) for x in (p, v, w, acc_b, target))
        self._update(p, v, q_xyzw, w, target)
        obs = self.observation(p, v, q_xyzw, w, acc_b, target)
        a = self.session.run(None, {"obs": obs[None]})[0][0]
        self.u = np.clip(F_HOVER + a * (F_MAX - F_HOVER), self.floor, F_MAX)
        self.u_lpf = lpf(self.u_lpf, self.u, 0.025)
        return a

    def track(self, p, v, q_xyzw, w, target, u_applied):
        """Shadow mode: while another controller (PX4) flies, call this every 10 ms with the rotor thrusts [N] it
        actually applies. It keeps the filters and the high-level state current, so the policy can take over on
        fault detection without a cold start (a cold start 80 ms into a failure crashes; see policies/README.md)."""
        p, v, w, target = (np.asarray(x, float) for x in (p, v, w, target))
        self._update(p, v, q_xyzw, w, target)
        self.u = np.clip(np.asarray(u_applied, float), 0.0, F_MAX)
        self.u_lpf = lpf(self.u_lpf, self.u, 0.025)

    def step(self, p, v, q_xyzw, w, acc_b, target):
        """Rotor thrusts [N], PX4 order: 0 front-right CCW, 1 back-left CCW, 2 front-left CW, 3 back-right CW."""
        self.action(p, v, q_xyzw, w, acc_b, target)
        return self.u.copy()

    @staticmethod
    def rotor_speed(thrust):
        """Rotor speed [rad/s] that gives this thrust in Gazebo (F = motorConstant * omega^2)."""
        return np.sqrt(np.maximum(thrust, 0.0) / MOTOR_CONSTANT)
