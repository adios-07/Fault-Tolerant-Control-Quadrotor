"""Rotor-failure environment on crazyflow, shared by train.py and eval.py (one module = train/eval parity).

Drone: crazyflow x500v2 (Holybro X500 v2, matched to our Gazebo custom_x500: 1.40 kg, 0-11.8 N per rotor,
T/W 3.44, PX4 motor order; see [x500v2] in crazyflow/dynamics/first_principles/params.toml) with first_principles
dynamics. The v1 runs (smoke1 ... i008) used DRONE = Drone.hb_x500 ("hb_x500-FTC": 2.28 kg, 0-17 N, old order).

Interface after Chen et al. 2025, "Learning-Based Passive Fault-Tolerant Control of a Quadrotor with Rotor
Failure" (arXiv 2503.02649, github.com/HITSZcjh/uav_ftc):
  high-level PID position + primary-axis attitude controller (ported from uav_ftc
  validation/high_level_controller.py) -> (p_des, q_des, a_z_des) -> RL policy -> 4 rotor thrusts.
Failure: from step t_fail, rotor i's thrust and torque curves are scaled by k_i (0 = dead). The policy never
observes k; only the reward uses it (privileged training signal, as in the paper).
All arrays are batched over worlds: (N, ...).
"""
import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.drones import Drone
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.pipeline import insert_fn_before, prepend_fn

FREQ, CTRL_HZ = 500, 100  # physics Hz, policy Hz
SUB, DT = FREQ // CTRL_HZ, 1.0 / CTRL_HZ
T_FAIL, T_END = 3.0, 15.0  # s, eval failure time and episode length
TARGET = jnp.array([0.0, 0.0, 1.5])
DRONE = Drone.x500v2  # Drone.hb_x500 reproduces the v1 runs
_P = load_params(Dynamics.first_principles, DRONE)
MASS, G = float(_P["mass"]), 9.81
F_MAX = float(_P["thrust_max"])  # N per rotor
F_HOVER = MASS * G / 4  # 3.43 N for x500v2
RPM2THRUST = jnp.asarray(_P["rpm2thrust"])  # nominal thrust curve: what the controller believes
RPM2TORQUE = jnp.asarray(_P["rpm2torque"])
REWARD = "paper"  # "smooth" = IDEA-009, "precise" = i019 (set by train.py --reward; training-only)
TILT_PEN = 0.0  # set by train.py --tilt-pen (training-only)
C_YAW = 40.0  # rad/s, spin allowed after a complete failure before it is penalised (hover needs ~15-30)
GRAV = jnp.array([0.0, 0.0, -G])
OBS_DIM, ACT_DIM = 27, 4

# Gazebo-gap model (make_sim(gz=True)): the differences the PX4/Gazebo bench found between Gazebo and this sim.
MOTOR_CONSTANT = 1.18e-5  # N/(rad/s)^2, Gazebo custom_x500: rotor speed omega = sqrt(thrust / motorConstant)
ROTOR_DRAG = 8.06428e-5  # Gazebo rotorDragCoefficient: per rotor F = -|omega| * c * v_perp (v at the rotor)
_mix = jnp.asarray(_P["mixing_matrix"])
ROTOR_POS = jnp.stack([-_mix[1] * _P["L"], _mix[0] * _P["L"], jnp.full(4, 0.069)], -1)  # (4, 3) body frame
F_FLOOR = 0.27  # N, PX4 armed-motor floor (150 rad/s) on rotor commands
DELAY_MAX = 10  # physics steps (2 ms each): the largest actuation delay train.py --gz-rand draws
# Buffer lengths: copied every step, so keep them as short as training needs (stress.py enlarges them before make_sim)
QUEUE_LEN = 20  # physics steps the actuation-delay queue holds (40 ms)
HIST_LEN = 4  # policy steps of true state kept for the estimate lag (30 ms)
# EKF-like estimate noise at noise scale 1: p [m], v [m/s], quaternion components (~1 deg), w [rad/s], acc_b [m/s^2]
EST_SIGMA = jnp.array([0.02] * 3 + [0.05] * 3 + [0.0087] * 4 + [0.05] * 3 + [0.1] * 3)
PROP_J0 = float(np.ravel(_P["prop_inertia"])[0])  # kg m^2, physical rotor inertia (Gazebo rotor izz)
EKF_TAU = 1.0  # s, how fast GPS fusion pulls the EKF velocity error back (EKF error model, see step)
# prop_j: rotor inertia x the physical value. Gazebo spins the rotor links at 1/rotorVelocitySlowdownSim = 1/10 of the
# rotor speed, so its gyroscopic and spin-up torques are 0.1x; that gap is what made i018 orbit on the bench.
# spin: the failed rotor keeps turning (Gazebo "prop" fault: thrust and yaw torque gone, inertia and drag stay).
# att_max / att_rate: EKF attitude error after the fault, ramping at att_rate [rad/s] up to att_max [rad].
NOMINAL = {"delay": 0, "drag_c": 1.0, "lag": 0, "noise": 0.0, "floor": 0.0, "prop_j": 1.0, "spin": 0.0,
           "att_max": 0.0, "att_rate": 0.0, "vel": 0.0, "wz": 0.0, "yaw": 0.0}  # what reset_where uses by default
# vel [m/s] (3 or (N,3)), wz [rad/s], yaw [rad]: flight state at the start (the bench faults in forward/climb/yaw flight)
# Plant-only "hardware" deviations (held-out test set in bench_sim.py, never trained on unless train.py --hw-rand):
# mass, J, motor_tau: x nominal; t_err: (4,) or (N,4) per-motor thrust/torque curve x; sag: max thrust x;
# cg: (3,) or (N,3) centre-of-mass offset in the body frame [m] (gz only, a torque from the thrust).
NOMINAL |= {"mass": 1.0, "J": 1.0, "motor_tau": 1.0, "t_err": 1.0, "sag": 1.0, "cg": 0.0}
# Bench onboard estimator (IDEA-022 B: gyro attitude + position/velocity KF, code/px4_ws/src/rotor_fail/src/pos_kf.py).
# kf: 1 = the controller sees it (from the fault on); head: constant heading offset [rad]; att_seed: (3,) or (N,3)
# attitude error at the fault [rad, world frame]; g_s / g_b: (3,) or (N,3) gyro scale error (fraction) and bias
# [rad/s] per body axis, integrated from the fault; gps_period / gps_delay: policy steps; gps_vn / gps_pj: GPS
# velocity / position noise [m/s, m]. The attitude error drifts whenever g_s / g_b are set, even with kf 0.
NOMINAL |= {"kf": 0.0, "head": 0.0, "att_seed": 0.0, "g_s": 0.0, "g_b": 0.0, "gps_period": 1, "gps_delay": 0,
            "gps_vn": 0.0, "gps_pj": 0.0}
KF_ACC, KF_POS, KF_VEL = 1.0, 0.5, 0.2  # the bench KF's sigmas (pos_kf.py)
GPS_HIST = 21  # policy steps of true p, v kept for the GPS delay (up to 200 ms)
# Stress-test disturbances (gz only unless noted; all off by default): wind [m/s, world, (3,) or (N,3)] plus a gust
# vector at 0.25 Hz; tilt: (4,3) or (N,4,3) thrust-direction offset per rotor in the body frame (bent mount, ~rad);
# q_ratio: the faulted rotor's drag torque x its thrust share (chipped blade), any mode; ramp: fault onset [steps].
NOMINAL |= {"wind": 0.0, "gust": 0.0, "tilt": 0.0, "q_ratio": 1.0, "ramp": 0}
GZ_EVAL = NOMINAL | {"delay": 5, "lag": 1, "noise": 0.5, "floor": F_FLOOR, "prop_j": 0.1}  # eval.py --gz


class Env(NamedTuple):
    data: object  # crazyflow SimData
    nom_t: jax.Array  # (N,1,4,3) healthy rpm2thrust
    nom_q: jax.Array  # (N,1,4,3) healthy rpm2torque
    t: jax.Array  # (N,) policy steps since reset
    t_fail: jax.Array  # (N,) step the fault starts
    k: jax.Array  # (N,4) rotor health after the fault, never observed
    u: jax.Array  # (N,4) last commanded thrust, N
    u_lpf: jax.Array  # (N,4) low-passed command (uav_ftc: tau 0.025 s)
    v_prev: jax.Array  # (N,3) for the accelerometer estimate
    acc_b: jax.Array  # (N,3) specific force in body frame (what an IMU measures)
    w_prev: jax.Array  # (N,3)
    dw_lpf: jax.Array  # (N,3) filtered angular acceleration (uav_ftc: tau 0.12 s)
    n_last: jax.Array  # (N,3) last desired primary axis (high-level controller state)
    hl_lpf: jax.Array  # (N,3) low-passed [p_des, q_des, a_z_des]
    est: jax.Array  # (N,13) state estimate [p, v, q_xyzw, w] the controller sees (= true state when nominal)
    hist: jax.Array  # (N,3,13) last true states, newest first, for the estimate lag
    lag: jax.Array  # (N,) estimate lag in policy steps (0-2)
    noise: jax.Array  # (N,) estimate noise scale (x EST_SIGMA)
    floor: jax.Array  # (N,) minimum rotor command, N
    key: jax.Array  # PRNG key for the estimate noise
    spin: jax.Array  # (N,) 1 = the failed rotor keeps turning (prop loss), 0 = it stops
    att_err: jax.Array  # (N,3) EKF attitude error at its full size: direction * att_max, rad (world frame)
    att_rate: jax.Array  # (N,) rad/s ramp of that error after the fault
    v_err: jax.Array  # (N,3) EKF velocity error that the attitude error integrates into
    sag: jax.Array  # (N,) plant max thrust x F_MAX (battery sag); the controller still believes F_MAX
    q_err: jax.Array  # (N,4) onboard attitude error, xyzw, world frame: estimate = q_err * true
    att_seed: jax.Array  # (N,3) rotation vector composed onto q_err at the fault
    g_s: jax.Array  # (N,3) gyro scale error
    g_b: jax.Array  # (N,3) gyro bias, rad/s
    kf: jax.Array  # (N,) 1 = the controller sees the onboard estimate
    kf_x: jax.Array  # (N,2,3) KF state [p; v]
    kf_P: jax.Array  # (N,2,2) KF covariance, the same for every axis
    gps: jax.Array  # (N,4) [period, delay, vel noise, pos noise]
    gps_hist: jax.Array  # (N,GPS_HIST,6) true [p, v], newest first
    q_ratio: jax.Array  # (N,) faulted rotor's torque share / thrust share
    ramp: jax.Array  # (N,) fault onset duration, policy steps (0 = instant)


def make_sim(n: int, control: Control = Control.rotor_vel, gz: bool = False, delay: bool = True) -> Sim:
    """crazyflow sim with per-world, per-motor thrust/torque curves so each world can have its own fault.
    gz=True adds the Gazebo-gap plugins at the 500 Hz physics rate: actuation delay and speed-dependent rotor drag
    (which replaces the fixed, hover-linearised drag matrix)."""
    sim = Sim(n_worlds=n, drone=DRONE, dynamics=Dynamics.first_principles, control=control, freq=FREQ)
    p = sim.data.params
    p = p.replace(rpm2thrust=jnp.broadcast_to(p.rpm2thrust, (n, 1, 4, 3)),
                  rpm2torque=jnp.broadcast_to(p.rpm2torque, (n, 1, 4, 3)),
                  prop_inertia=jnp.broadcast_to(p.prop_inertia, (n, 1, 1)), mass=jnp.broadcast_to(p.mass, (n, 1, 1)),
                  J=jnp.broadcast_to(p.J, (n, 1, 3, 3)), J_inv=jnp.broadcast_to(p.J_inv, (n, 1, 3, 3)),
                  rotor_dyn_coef=jnp.broadcast_to(p.rotor_dyn_coef, (n, 1, 1, 4)))
    plugins = {}
    if gz:
        p = p.replace(drag_matrix=jnp.zeros_like(p.drag_matrix))
        zeros = jnp.zeros((n, 1, 4))
        plugins = {"cmd": zeros, "queue": jnp.zeros((QUEUE_LEN + 1, n, 1, 4)),
                   "delay": jnp.zeros(n, jnp.int32), "drag_c": jnp.ones(n), "cg": jnp.zeros((n, 3)),
                   "wind": jnp.zeros((n, 3)), "wind0": jnp.zeros((n, 3)), "gust": jnp.zeros((n, 3)),
                   "tilt": jnp.zeros((n, 4, 3))}
        if delay:  # off only for controllers that write rotor commands inside the pipeline (Mellinger)
            prepend_fn(sim.step_pipeline, _actuation_delay)
        insert_fn_before(sim.step_pipeline, "integration", _rotor_drag)
    sim.data = sim.data.replace(params=p, plugins=sim.data.plugins | plugins)
    sim.build_default_data()
    return sim


def _actuation_delay(data):
    """gz plugin: the rotors get the command issued `delay` physics steps ago (ROS -> PX4 -> Gazebo latency)."""
    pl = data.plugins
    queue = jnp.concatenate([pl["cmd"][None], pl["queue"][:-1]])  # newest first
    cmd = queue[pl["delay"], jnp.arange(queue.shape[1])]
    return data.replace(controls=data.controls.replace(rotor_vel=cmd), plugins=pl | {"queue": queue})


def _rotor_drag(data):
    """gz plugin: Gazebo MulticopterMotorModel rotor drag, F_i = -|omega_i| * c * v_perp_i, with v_i the velocity of
    rotor i (body velocity + w x r_i) without its axial part. The w x r_i part damps the yaw spin."""
    s, p = data.states, data.params
    R = rotmat(s.quat[:, 0])
    v_b = jnp.einsum("nji,nj->ni", R, s.vel[:, 0] - data.plugins["wind"])  # air-relative
    v_rotor = v_b[:, None] + jnp.cross(s.ang_vel[:, 0][:, None], ROTOR_POS[None])  # (N, 4, 3)
    v_perp = v_rotor.at[..., 2].set(0.0)
    r = s.rotor_vel[:, 0]
    # Gazebo's drag uses the rotor speed, so a rotor that lost its prop still drags; a stopped one spins down
    omega = jnp.sqrt(jnp.maximum(RPM2THRUST[0] + RPM2THRUST[1] * r + RPM2THRUST[2] * r ** 2, 0.0) / MOTOR_CONSTANT)
    f = -(omega * data.plugins["drag_c"][:, None] * ROTOR_DRAG)[..., None] * v_perp
    force = jnp.einsum("nij,nj->ni", R, f.sum(1))
    fi = p.rpm2thrust[:, 0, :, 0] + p.rpm2thrust[:, 0, :, 1] * r + p.rpm2thrust[:, 0, :, 2] * r ** 2  # (N, 4)
    f = f + fi[..., None] * data.plugins["tilt"]  # tilted thrust axis: the sideways part of each rotor's thrust
    force = jnp.einsum("nij,nj->ni", R, f.sum(1))
    torque = jnp.cross(ROTOR_POS[None], f).sum(1)
    # CG offset: the rotor thrust acts at -cg from the centre of mass
    thrust = fi.sum(-1)
    torque = torque + jnp.cross(-data.plugins["cg"], jnp.zeros_like(torque).at[:, 2].set(thrust))
    torque = jnp.einsum("nij,nj->ni", R, torque)  # world frame, as crazyflow wants
    return data.replace(states=s.replace(force=force[:, None], torque=torque[:, None]))


def apply_fault(data, nom_t, nom_q, k, kq=None):
    """Scale each rotor's thrust and torque curve by its health k (N,4): 0 = complete failure. kq: torque scale
    when it differs from the thrust scale."""
    kk = k[:, None, :, None]
    kq = kk if kq is None else kq[:, None, :, None]
    return data.replace(params=data.params.replace(rpm2thrust=nom_t * kk, rpm2torque=nom_q * kq))


def rotmat(q):
    """Body-to-world rotation from xyzw quaternions (crazyflow/scipy order)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack([
        jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def quat_mul(a, b):
    """Hamilton product of xyzw quaternions: rotation b, then a."""
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return jnp.concatenate([aw * bv + bw * av + jnp.cross(av, bv), aw * bw - jnp.sum(av * bv, -1, keepdims=True)], -1)


def expmap(r):
    """Rotation vector (..., 3) -> xyzw quaternion."""
    half = 0.5 * jnp.linalg.norm(r, axis=-1, keepdims=True)
    return jnp.concatenate([r * jnp.sinc(half / jnp.pi) * 0.5, jnp.cos(half)], -1)  # sin(h)/(2h) * r = axis sin(h)


def lpf(prev, x, tau):
    return (DT * x + tau * prev) / (tau + DT)


def high_level(p, v, R, wz, n_last, hl_lpf):
    """uav_ftc PositionController + PrimaryAxisAttitudeController (same gains), batched, run at 100 Hz.

    Returns the new n_last, hl_lpf state and the clipped command [p_des, q_des, a_z_des].
    """
    vel_t = jnp.clip(TARGET - p, -10.0, 10.0)  # kp_pos = 1
    acc = jnp.array([2.0, 2.0, 6.0]) * (vel_t - v)  # kp_vel
    max_lat = G * jnp.tan(10.0 / 57.3)  # 10 deg max tilt command
    ratio = jnp.maximum(jnp.linalg.norm(acc[:, :2], axis=-1, keepdims=True) / max_lat, 1.0)
    acc = jnp.concatenate([acc[:, :2] / ratio, jnp.clip(acc[:, 2:], -5.0, 5.0)], -1)
    thrust_dir = acc - GRAV
    n_des = thrust_dir / jnp.linalg.norm(thrust_dir, axis=-1, keepdims=True)
    n_dot = jnp.clip((n_des - n_last) / DT, -0.5, 0.5)

    h = jnp.einsum("nji,nj->ni", R, n_des)  # desired primary axis in body frame (R^T n_des)
    h3 = jnp.maximum(h[:, 2], 0.1)  # guard: original divides by h3, which blows up when tumbling
    vout = -5.0 * h[:, :2]  # kx, ky = 5, body primary axis n_B = (0, 0, 1)
    nd = jnp.einsum("nji,nj->ni", R, n_dot)[:, :2]
    tmp = vout - wz[:, None] * jnp.stack([h[:, 1], -h[:, 0]], -1) - nd
    pq = jnp.stack([tmp[:, 1] / h3, -tmp[:, 0] / h3], -1)
    az = jnp.linalg.norm(thrust_dir, axis=-1)
    hl_lpf = lpf(hl_lpf, jnp.concatenate([pq, az[:, None]], -1), 1.0 / 50)  # BW = 50
    cmd = jnp.concatenate([jnp.clip(hl_lpf[:, :2], -10, 10), jnp.clip(hl_lpf[:, 2:], -50, 50)], -1)
    return n_des, hl_lpf, cmd


def observe(e: Env):
    """27-d policy input, uav_ftc layout and scaling. Onboard-type signals only: no k, no rotor_vel, no force.
    Uses the state estimate e.est (the true state unless the gz lag/noise is on)."""
    p, v, q, w = e.est[:, :3], e.est[:, 3:6], e.est[:, 6:10], e.est[:, 10:]
    q = q * jnp.sign(q[:, 3:4] + 1e-9)  # q and -q are the same attitude
    hl = jnp.concatenate([jnp.clip(e.hl_lpf[:, :2], -10, 10), jnp.clip(e.hl_lpf[:, 2:], -50, 50)], -1)
    return jnp.concatenate([
        (p - TARGET) / 5, v / 5,
        q[:, [3, 0, 1, 2]],  # wxyz as uav_ftc
        w / jnp.array([5.0, 5.0, 20.0]),  # wz / 20: spin of 15-40 rad/s stays in range
        e.u_lpf / 10, e.acc_b / jnp.array([5.0, 5.0, 20.0]),
        hl[:, :2] / 5, jnp.zeros_like(hl[:, :1]), hl[:, 2:] / 20,  # [p_des, q_des, wz_cmd = 0, a_z_des]
        e.dw_lpf / 50], -1)


def init(sim: Sim, seed: int = 0) -> Env:
    n = sim.n_worlds
    z3, z4, zn = jnp.zeros((n, 3)), jnp.zeros((n, 4)), jnp.zeros(n)
    return Env(sim.data, sim.data.params.rpm2thrust, sim.data.params.rpm2torque, jnp.zeros(n, jnp.int32),
               jnp.zeros(n, jnp.int32), jnp.ones((n, 4)), z4, z4, z3, z3, z3, z3, z3, z3,
               jnp.zeros((n, 13)), jnp.zeros((n, HIST_LEN, 13)), jnp.zeros(n, jnp.int32), zn, zn, jax.random.key(seed),
               zn, z3, zn, z3, jnp.ones(n), q_err=jnp.zeros((n, 4)).at[:, 3].set(1.0), att_seed=z3, g_s=z3, g_b=z3,
               kf=zn, kf_x=jnp.zeros((n, 2, 3)), kf_P=jnp.zeros((n, 2, 2)), gps=jnp.zeros((n, 4)).at[:, 0].set(1.0),
               gps_hist=jnp.zeros((n, GPS_HIST, 6)), q_ratio=jnp.ones(n), ramp=jnp.zeros(n))


def reset_where(e: Env, mask, pos, t_fail, k, rand=None) -> Env:
    """Reset worlds where mask (N,) is True: hover at pos with rotors already at hover speed (crazyflow resets
    them to 0 RPM, which drops the drone ~1.8 m) and schedule fault k (N,4) at policy step t_fail (N,).
    rand: per-world Gazebo-gap settings (keys of NOMINAL, scalars or (N,) arrays); default NOMINAL."""
    rand = NOMINAL | (rand or {})
    n = mask.shape[0]
    full = lambda x: jnp.broadcast_to(jnp.asarray(x), (n,))  # noqa: E731
    s = e.data.states
    m3, m4 = mask[:, None, None], mask[:, None]
    hover_rpm = motor_force2rotor_vel(jnp.full((4,), F_HOVER), RPM2THRUST)
    vel = jnp.broadcast_to(jnp.asarray(rand["vel"]), (n, 3))
    w0 = jnp.zeros((n, 3)).at[:, 2].set(full(rand["wz"]))
    yaw = full(rand["yaw"])
    quat = jnp.stack([jnp.zeros(n), jnp.zeros(n), jnp.sin(yaw / 2), jnp.cos(yaw / 2)], -1)
    states = s.replace(
        pos=jnp.where(m3, pos[:, None], s.pos), vel=jnp.where(m3, vel[:, None], s.vel),
        quat=jnp.where(m3, quat[:, None], s.quat), ang_vel=jnp.where(m3, w0[:, None], s.ang_vel),
        force=jnp.where(m3, 0.0, s.force), torque=jnp.where(m3, 0.0, s.torque),
        rotor_vel=jnp.where(m3, hover_rpm, s.rotor_vel))
    hover_u = jnp.full((4,), F_HOVER)
    true = jnp.concatenate([pos, vel, quat, w0], -1)
    plugins = e.data.plugins
    if "queue" in plugins:
        plugins = plugins | {
            "cmd": jnp.where(m3, hover_rpm, plugins["cmd"]),
            "queue": jnp.where(m3[None], hover_rpm, plugins["queue"]),
            "delay": jnp.where(mask, full(rand["delay"]).astype(jnp.int32), plugins["delay"]),
            "drag_c": jnp.where(mask, full(rand["drag_c"]), plugins["drag_c"]),
            "cg": jnp.where(m4, jnp.broadcast_to(jnp.asarray(rand["cg"]), (n, 3)), plugins["cg"]),
            "wind0": jnp.where(m4, jnp.broadcast_to(jnp.asarray(rand["wind"]), (n, 3)), plugins["wind0"]),
            "gust": jnp.where(m4, jnp.broadcast_to(jnp.asarray(rand["gust"]), (n, 3)), plugins["gust"]),
            "tilt": jnp.where(m3, jnp.broadcast_to(jnp.asarray(rand["tilt"]), (n, 4, 3)), plugins["tilt"])}
    params = e.data.params
    m5 = mask[:, None, None, None]
    J0, dyn0 = jnp.asarray(_P["J"]), jnp.asarray(_P["rotor_dyn_coef"])
    Jinv0 = jnp.linalg.inv(J0)
    params = params.replace(
        prop_inertia=jnp.where(m3, (full(rand["prop_j"]) * PROP_J0)[:, None, None], params.prop_inertia),
        mass=jnp.where(m3, (full(rand["mass"]) * MASS)[:, None, None], params.mass),
        J=jnp.where(m5, full(rand["J"])[:, None, None, None] * J0, params.J),
        J_inv=jnp.where(m5, Jinv0 / full(rand["J"])[:, None, None, None], params.J_inv),
        rotor_dyn_coef=jnp.where(m5, dyn0 / full(rand["motor_tau"])[:, None, None, None], params.rotor_dyn_coef))
    t_err = jnp.broadcast_to(jnp.asarray(rand["t_err"]), (n, 4))[:, None, :, None]
    nom_t = jnp.where(m5, RPM2THRUST * t_err, e.nom_t)
    nom_q = jnp.where(m5, RPM2TORQUE * t_err, e.nom_q)
    key, k_dir = jax.random.split(e.key)
    d = jax.random.normal(k_dir, (n, 3))
    att_err = d / jnp.linalg.norm(d, axis=-1, keepdims=True) * full(rand["att_max"])[:, None]  # random axis
    v3 = lambda x: jnp.broadcast_to(jnp.asarray(x, jnp.float32), (n, 3))  # noqa: E731
    head = full(rand["head"])
    q_head = jnp.stack([jnp.zeros(n), jnp.zeros(n), jnp.sin(head / 2), jnp.cos(head / 2)], -1)
    gps = jnp.stack([full(rand["gps_period"]), full(rand["gps_delay"]), full(rand["gps_vn"]), full(rand["gps_pj"])], -1)
    return e._replace(
        data=e.data.replace(states=states, plugins=plugins, params=params), key=key, nom_t=nom_t, nom_q=nom_q,
        sag=jnp.where(mask, full(rand["sag"]), e.sag), q_err=jnp.where(m4, q_head, e.q_err),
        att_seed=jnp.where(m4, v3(rand["att_seed"]), e.att_seed), g_s=jnp.where(m4, v3(rand["g_s"]), e.g_s),
        g_b=jnp.where(m4, v3(rand["g_b"]), e.g_b), kf=jnp.where(mask, full(rand["kf"]), e.kf),
        gps=jnp.where(m4, gps, e.gps), q_ratio=jnp.where(mask, full(rand["q_ratio"]), e.q_ratio),
        ramp=jnp.where(mask, full(rand["ramp"]).astype(jnp.float32), e.ramp), gps_hist=jnp.where(m3, jnp.concatenate([pos, vel], -1)[:, None], e.gps_hist),
        spin=jnp.where(mask, full(rand["spin"]), e.spin), att_err=jnp.where(m4, att_err, e.att_err),
        att_rate=jnp.where(mask, full(rand["att_rate"]), e.att_rate), v_err=jnp.where(m4, 0.0, e.v_err),
        est=jnp.where(m4, true, e.est), hist=jnp.where(m3, true[:, None], e.hist),
        lag=jnp.where(mask, full(rand["lag"]).astype(jnp.int32), e.lag),
        noise=jnp.where(mask, full(rand["noise"]), e.noise), floor=jnp.where(mask, full(rand["floor"]), e.floor),
        t=jnp.where(mask, 0, e.t), t_fail=jnp.where(mask, t_fail, e.t_fail), k=jnp.where(m4, k, e.k),
        u=jnp.where(m4, hover_u, e.u), u_lpf=jnp.where(m4, hover_u, e.u_lpf),
        v_prev=jnp.where(m4, vel, e.v_prev), acc_b=jnp.where(m4, jnp.array([0.0, 0.0, G]), e.acc_b),
        w_prev=jnp.where(m4, w0, e.w_prev), dw_lpf=jnp.where(m4, 0.0, e.dw_lpf),
        n_last=jnp.where(m4, jnp.array([0.0, 0.0, 1.0]), e.n_last),
        hl_lpf=jnp.where(m4, jnp.array([0.0, 0.0, G]), e.hl_lpf))


def step(sim_step, e: Env, action):
    """One 100 Hz policy step. action in [-1, 1]^4, 0 = hover thrust. Returns (env, reward, crashed)."""
    u = jnp.clip(F_HOVER + action * (F_MAX - F_HOVER), e.floor[:, None], F_MAX)
    onset = jnp.clip((e.t - e.t_fail + 1) / jnp.maximum(e.ramp, 1.0), 0.0, 1.0)[:, None]  # 1 at t_fail when ramp 0
    k_now = jnp.where((e.t >= e.t_fail)[:, None], 1.0 - (1.0 - e.k) * onset, 1.0)
    kq = jnp.where(e.k < 1.0, k_now * e.q_ratio[:, None], k_now)
    data = apply_fault(e.data, e.nom_t, e.nom_q, k_now, kq)
    # healthy curve: the controller does not know the fault; a failed rotor stops (it has inertia now)
    rpm = motor_force2rotor_vel(jnp.minimum(u, e.sag[:, None] * F_MAX), RPM2THRUST) * ((k_now > 0.0) | (e.spin[:, None] > 0))  # prop loss: still turns
    if "queue" in data.plugins:  # gz: the actuation-delay plugin feeds the rotors
        gust = data.plugins["gust"] * jnp.sin(2 * jnp.pi * 0.25 * DT * e.t)[:, None]
        data = data.replace(plugins=data.plugins | {"cmd": rpm[:, None], "wind": data.plugins["wind0"] + gust})
    else:
        data = data.replace(controls=data.controls.replace(rotor_vel=rpm[:, None]))
    data = sim_step(data, SUB)

    s = data.states
    p, v, w = s.pos[:, 0], s.vel[:, 0], s.ang_vel[:, 0]
    R = rotmat(s.quat[:, 0])
    key, k_noise = jax.random.split(e.key)
    eps = jax.random.normal(k_noise, (p.shape[0], 16)) * EST_SIGMA * e.noise[:, None]
    acc_b = jnp.einsum("nji,nj->ni", R, (v - e.v_prev) / DT - GRAV) + eps[:, 13:]  # IMU: finite-diff velocity
    # state estimate: the true state `lag` policy steps ago, plus noise (EKF-like); exact when nominal
    hist = jnp.concatenate([jnp.concatenate([p, v, s.quat[:, 0], w], -1)[:, None], e.hist[:, :HIST_LEN - 1]], 1)
    est = jnp.take_along_axis(hist, e.lag[:, None, None], 1)[:, 0] + eps[:, :13]
    # EKF error model (from the bench's EKF2 logs): after the fault the attitude estimate drifts by a rotation that
    # ramps up to att_err, and the velocity estimate integrates the thrust direction it gets wrong (GPS pulls it back)
    size = jnp.linalg.norm(e.att_err, axis=-1)
    ramp = jnp.clip(e.att_rate * (e.t + 1 - e.t_fail) * DT / jnp.maximum(size, 1e-6), 0.0, 1.0)
    dth = e.att_err * ramp[:, None]
    v_err = e.v_err + DT * (G * jnp.cross(dth, jnp.array([0.0, 0.0, 1.0])) - e.v_err / EKF_TAU)
    q_ekf = expmap(dth)
    est = est.at[:, 6:10].set(quat_mul(q_ekf, est[:, 6:10] / jnp.linalg.norm(est[:, 6:10], axis=-1, keepdims=True)))
    est = est.at[:, 3:6].add(v_err)
    # Onboard estimator (bench IDEA-022 B). Attitude: from the fault the raw gyro (1 + g_s) w + g_b is integrated, so
    # the world-frame error obeys d q_err / dt = 1/2 (R_est dw) * q_err exactly, dw = g_s w + g_b.
    q_t = s.quat[:, 0]
    seed = e.t == e.t_fail
    q_err = jnp.where(seed[:, None], quat_mul(expmap(e.att_seed), e.q_err), e.q_err)
    dw = jnp.einsum("nij,nj->ni", rotmat(quat_mul(q_err, q_t)), e.g_s * w + e.g_b)
    q_err = jnp.where((e.t >= e.t_fail)[:, None], quat_mul(expmap(dw * DT), q_err), q_err)
    q_err = q_err / jnp.linalg.norm(q_err, axis=-1, keepdims=True)
    # Position / velocity: pos_kf.PosVelKF, started from truth at the fault, predicted with the specific force rotated by
    # the estimated attitude, corrected by delayed, noisy GPS fixes every gps_period steps.
    gps_hist = jnp.concatenate([jnp.concatenate([p, v], -1)[:, None], e.gps_hist[:, :-1]], 1)
    R_est = rotmat(quat_mul(q_err, q_t))
    acc_w = jnp.einsum("nij,nj->ni", R_est, acc_b) + GRAV
    x = jnp.where(seed[:, None, None], jnp.stack([p, v], 1), e.kf_x)
    P = jnp.where(seed[:, None, None], jnp.diag(jnp.array([0.1, 0.05])), e.kf_P)
    x = x + DT * jnp.stack([x[:, 1] + 0.5 * DT * acc_w, acc_w], 1)
    F = jnp.array([[1.0, DT], [0.0, 1.0]])
    P = F @ P @ F.T + KF_ACC ** 2 * jnp.array([[DT ** 4 / 4, DT ** 3 / 2], [DT ** 3 / 2, DT ** 2]])
    fix = (e.t - e.t_fail) % e.gps[:, 0].astype(jnp.int32) == 0
    z = jnp.take_along_axis(gps_hist, e.gps[:, 1].astype(jnp.int32)[:, None, None], 1)[:, 0]
    key, k_gps = jax.random.split(key)
    z = z + jax.random.normal(k_gps, z.shape) * jnp.repeat(e.gps[:, [3, 2]], 3, -1)
    for i, sig in ((0, KF_POS), (1, KF_VEL)):  # sequential position then velocity update, as pos_kf.update_gps
        gain = P[:, :, i] / (P[:, i, i] + sig ** 2)[:, None]  # (N, 2)
        x_new = x + gain[:, :, None] * (z[:, None, 3 * i:3 * i + 3] - x[:, i:i + 1])
        P_new = P - gain[:, :, None] * P[:, i:i + 1, :]
        x, P = jnp.where(fix[:, None, None], x_new, x), jnp.where(fix[:, None, None], P_new, P)
    on = e.kf > 0
    q_on = quat_mul(q_err, est[:, 6:10] / jnp.linalg.norm(est[:, 6:10], axis=-1, keepdims=True))
    w_on = est[:, 10:] * (1.0 + e.g_s) + e.g_b
    pv = jnp.where((e.t >= e.t_fail)[:, None, None], x, est[:, None, :6].reshape(-1, 2, 3))
    est = jnp.where(on[:, None], jnp.concatenate([pv.reshape(-1, 6), q_on, w_on], -1), est)
    p_e, v_e, w_e = est[:, :3], est[:, 3:6], est[:, 10:]
    dw_lpf = (w_e - e.w_prev + 0.12 * e.dw_lpf) / (0.12 + DT)  # uav_ftc LPF.calc_derivative, tau 0.12 s
    n_last, hl_lpf, hl = high_level(p_e, v_e, rotmat(est[:, 6:10]), w_e[:, 2], e.n_last, e.hl_lpf)
    du = jnp.sum((u - e.u) ** 2, -1)  # oscillation term, before e.u is overwritten
    e = e._replace(data=data, t=e.t + 1, v_err=v_err, q_err=q_err, kf_x=x, kf_P=P, gps_hist=gps_hist, u=u, u_lpf=lpf(e.u_lpf, u, 0.025), v_prev=v, acc_b=acc_b, w_prev=w_e,
                   dw_lpf=dw_lpf, n_last=n_last, hl_lpf=hl_lpf, est=est, hist=hist, key=key)

    # Paper reward terms and weights (0.1, 0.01, 0.02, 0.0008, 0.2). Yaw: penalise spin while yaw is still
    # controllable (healthy, or mild fault), only spin beyond C_YAW after a severe fault.
    yaw_free = (e.t > e.t_fail) & (jnp.min(e.k, -1) < 0.5)
    wz_pen = jnp.where(yaw_free, jnp.maximum(jnp.abs(w[:, 2]) - C_YAW, 0.0) ** 2, w[:, 2] ** 2)
    penalty = (0.1 * jnp.sum((p - TARGET) ** 2, -1) + 0.01 * jnp.sum(v ** 2, -1)
               + 0.02 * ((w[:, 0] - hl[:, 0]) ** 2 + (w[:, 1] - hl[:, 1]) ** 2 + wz_pen)
               + 0.0008 * du
               + 0.2 * (acc_b[:, 2] - hl[:, 2]) ** 2)
    if REWARD == "precise":  # i019: the paper's 0.1|e|^2 barely costs a 0.3 m orbit (0.01); this adds up to 0.3 near the target
        penalty = penalty + 0.3 * (1.0 - jnp.exp(-jnp.sum((p - TARGET) ** 2, -1) / 0.25 ** 2))
    if TILT_PEN:  # i022: tilt beyond 70 deg costs up to TILT_PEN at 90 deg (the bench counts > 90 deg as a flip)
        penalty = penalty + TILT_PEN * jnp.maximum(0.342 - R[:, 2, 2], 0.0) / 0.342
    if REWARD == "smooth":
        # IDEA-009: while yaw is still controllable, spin costs 10x more (wz weight 0.2) and tilt is penalised, so
        # flying in failure mode on a healthy drone no longer pays; smooth saturation instead of a flat cap.
        penalty = penalty + jnp.where(yaw_free, 0.0, 0.18 * w[:, 2] ** 2 + 0.2 * (1.0 - R[:, 2, 2]))
        reward = 1.5 - penalty / (1.0 + penalty)  # in (0.5, 1.5]: still positive, so ending the episode never pays
    else:
        # Alive bonus 1.5 and penalty capped at 1: reward stays positive, so ending the episode never pays.
        reward = 1.5 - jnp.minimum(penalty, 1.0)
    crashed = is_crashed(p, R, w)
    return e, jnp.where(crashed, 0.0, reward), crashed


def is_crashed(p, R, w):
    """Floor, flew > 3 m away, upside down, spin beyond any IMU (80 rad/s), or NaN."""
    return ((p[:, 2] < 0.1) | (jnp.linalg.norm(p - TARGET, axis=-1) > 3.0) | (R[:, 2, 2] < 0.0)
            | (jnp.linalg.norm(w, axis=-1) > 80.0) | ~jnp.all(jnp.isfinite(p), -1))


# Symmetries of the drone (x/y mass and inertia equal, CoM on the axis). Mirror x -> -x maps each motor to the one
# at (-x, y) and reverses every spin, which must match that motor's spin; a 180 deg yaw rotation maps each motor to
# the one at (-x, -y) with the same spin. So a policy that handles "motor 0 failed" handles motor i after
# transforming its obs into the frame where motor i looks like motor 0. The motor maps come from the mixing matrix
# (row 0 = sign y, row 1 = -sign x, row 2 = spin), so they follow the drone's motor order.
def _signs(sx, sy):
    """Obs sign flips for the map diag(sx, sy, 1): vectors flip as (sx, sy, 1), pseudovectors (angular
    velocity, quaternion vector part, angular acceleration, body-rate commands) as det * (sx, sy, 1)."""
    v = jnp.array([sx, sy, 1.0])
    a = sx * sy * v
    return jnp.concatenate([v, v, jnp.ones(1), a, a, jnp.ones(4), v, a[:2], jnp.ones(2), a])


def _motor_maps(mix):
    """Motor permutations of the mirror and the 180 deg rotation, and which transforms bring motor i onto motor 0."""
    sy, sx, spin = mix[0], -mix[1], mix[2]
    find = lambda x, y: [int(((sx == x) & (sy == y)).argmax()) for x, y in zip(x, y)]  # noqa: E731
    mirror, rot = find(-sx, sy), find(-sx, -sy)
    assert all(spin[mirror] == -spin) and all(spin[rot] == spin), "motor layout is not mirror/rotation symmetric"
    to0 = {0: (), mirror[0]: ("m",), rot[0]: ("r",), mirror[rot[0]]: ("m", "r")}
    return mirror, rot, [to0[i] for i in range(4)]


_mirror, _rot, _to0 = _motor_maps(np.asarray(_P["mixing_matrix"]))
MIRROR = (_signs(-1.0, 1.0), jnp.array(_mirror))
ROT180 = (_signs(-1.0, -1.0), jnp.array(_rot))
TO_MOTOR0 = tuple(tuple({"m": MIRROR, "r": ROT180}[s] for s in seq) for seq in _to0)  # "motor i failed" -> motor 0


# PX4's reaction between the fault and the handover, from the bench's actuator_motors logs (8 runs, all identical up to
# symmetry): thrust [N] of [failed rotor, its diagonal partner, the other two] every 10 ms from the fault on. PX4's
# attitude loop drives the failed rotor to max and the partner to the floor; after 80 ms it stays saturated.
PX4_REACT = jnp.array([[3.45, 3.40, 3.42], [4.8, 2.3, 3.43], [7.4, 1.05, 3.48], [8.6, 0.75, 3.52], [9.4, 0.6, 3.56],
                       [10.3, 0.5, 3.6], [11.1, 0.43, 3.65], [11.7, 0.4, 3.7], [11.8, 0.27, 3.5]])


def px4_standin(e):
    """Action that reproduces PX4's rotor commands after the fault (stand-in for PX4 until the handover)."""
    row = PX4_REACT[jnp.clip(e.t - e.t_fail, 0, len(PX4_REACT) - 1)]  # (N, 3)
    failed = jnp.argmin(e.k, -1)
    f, o = jax.nn.one_hot(failed, 4), jax.nn.one_hot(ROT180[1][failed], 4)
    thrust = f * row[:, :1] + o * row[:, 1:2] + (1 - f - o) * row[:, 2:]
    return (thrust - F_HOVER) / (F_MAX - F_HOVER)


def obs_to_motor0(obs, i):
    for signs, perm in TO_MOTOR0[i]:
        obs = (obs * signs).at[:, 13:17].set((obs * signs)[:, 13:17][:, perm])  # u_prev is per motor
    return obs


def act_from_motor0(act, i):
    for _, perm in reversed(TO_MOTOR0[i]):  # each perm is its own inverse
        act = act[:, perm]
    return act

