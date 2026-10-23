#!/usr/bin/env python3
"""PPO for the passive fault-tolerant hover policy (one file, jax + flax + optax, PureJaxRL style).

The whole rollout and update run in one jitted function per iteration (no per-step host sync).
  python train.py --id smoke0 --steps 5e6 [--t-fail 0 0] [--motor 0] [--p-fault 1.0] [--partial 0.0]
Writes runs/<id>/policy.pkl (actor params) and runs/<id>/log.txt.
"""
import argparse
import pickle
import time
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import ftc


def mlp(x, out, scale, name=None):
    """3x64 tanh MLP (paper: 3 layers of 64)."""
    for j in range(3):
        x = nn.tanh(nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                             name=None if name is None else f"{name}_{j}")(x))
    return nn.Dense(out, kernel_init=nn.initializers.orthogonal(scale), name=None if name is None else f"{name}_out")(x)


class ActorCritic(nn.Module):
    """Actor, critic, learned state-independent std. Returns (mean, log_std, value, selector weights).

    selector=False: one MLP actor (weights fixed at 0.25, unused).
    selector=True: paper's Selector-Controller actor: 4 controller MLPs (one per failure case) mixed by a
    softmax selector MLP, mean = sum_i w_i u_i. The selector is supervised with the fault label in training only.
    """
    selector: bool = False

    @nn.compact
    def __call__(self, x):
        if self.selector:
            w = nn.softmax(mlp(x, 4, 0.01, "sel"))
            means = jnp.stack([mlp(x, ftc.ACT_DIM, 0.01, f"ctrl{i}") for i in range(4)], -2)
            mean = jnp.einsum("...i,...ia->...a", w, means)
        else:  # unnamed layers: keeps old policy.pkl files loadable
            w = jnp.full(x.shape[:-1] + (4,), 0.25)
            mean = mlp(x, ftc.ACT_DIM, 0.01)
        log_std = self.param("log_std", nn.initializers.constant(-1.0), (ftc.ACT_DIM,))
        value = mlp(x, 1, 1.0, "critic" if self.selector else None)[..., 0]
        return mean, log_std, value, w


def net_for(params):
    return ActorCritic(selector="sel_0" in params["params"])


class Selector(nn.Module):
    """Fault selector for the symmetric controller: obs -> softmax weights over 'motor i failed'
    (+ 'healthy' as mode 4 when a healthy expert is used, IDEA-013)."""
    modes: int = 4
    raw: bool = False  # logits instead of weights (equivariant selector, see mix_weights)

    @nn.compact
    def __call__(self, x):
        z = mlp(x, self.modes, 0.01)
        return z if self.raw else nn.softmax(z)


def n_modes(params):
    return 5 if "expert" in params else 4


def mix_weights(params, obs):
    """Selector weights over the modes. "selector_eq" (i025+) makes the whole mixture exactly equivariant: the score
    for "motor i failed" is the selector's motor-0-vs-healthy log-odds on obs_to_motor0(obs, i) alone, and "healthy"
    is the reference logit 0, so the weights permute exactly like the motors under the drone's symmetries."""
    if "selector_eq" not in params:
        return Selector(n_modes(params)).apply(params["selector"], obs)
    z = [Selector(5, raw=True).apply(params["selector_eq"], ftc.obs_to_motor0(obs, i)) for i in range(4)]
    return jax.nn.softmax(jnp.stack([zi[:, 0] - zi[:, 4] for zi in z] + [jnp.zeros(obs.shape[0])], -1))


def symmetrise(f, obs):
    """Average f's (action, value) over the 4 symmetries: f(T obs) mapped back. Exactly equivariant."""
    outs = [f(ftc.obs_to_motor0(obs, i)) for i in range(4)]
    return (sum(ftc.act_from_motor0(a, i) for i, (a, _) in enumerate(outs)) / 4, sum(v for _, v in outs) / 4)


def expert_out(params, obs):
    """Healthy expert's clipped action and value; symmetrised when the selector is equivariant."""
    def f(x):
        o = ActorCritic().apply(params["expert"], x)
        return jnp.clip(o[0], -1.0, 1.0), o[2]
    return symmetrise(f, obs) if "selector_eq" in params else f(obs)


def symmetric_acts(params, obs):
    """Actions of the motor-0 base under the 4 symmetries (+ the healthy expert's action): (N, modes, 4)."""
    acts = [ftc.act_from_motor0(policy_mean(params["base"], ftc.obs_to_motor0(obs, i)), i) for i in range(4)]
    if "expert" in params:
        acts.append(expert_out(params, obs)[0])
    return jnp.stack(acts, 1)


def symmetric_mean(params, obs):
    """Symmetric Selector-Controller: 4 controllers = the motor-0 policy under the drone's symmetries
    (ftc.TO_MOTOR0), mixed by the learned selector. Needs no fault information at run time."""
    w = mix_weights(params, obs)
    return jnp.einsum("ni,nia->na", w, symmetric_acts(params, obs)), w


def policy_mean(params, obs):
    """Deterministic action used at evaluation / deployment."""
    if "base" in params:
        return symmetric_mean(params, obs)[0]
    return jnp.clip(net_for(params).apply(params, obs)[0], -1.0, 1.0)


class Joint:
    """IDEA-012: the symmetric Selector-Controller as one PPO actor-critic, so PPO trains the shared motor-0 base
    on the states the deployed mixture visits. Mean and value are selector-weighted mixes of the base under the 4
    symmetries; the selector is trained only by its supervised loss (stop_gradient), log_std is the base's."""

    def apply(self, params, obs):
        lead, obs = obs.shape[:-1], obs.reshape(-1, obs.shape[-1])  # the symmetry maps take (N, obs)
        outs = [ActorCritic().apply(params["base"], ftc.obs_to_motor0(obs, i)) for i in range(4)]
        acts = [ftc.act_from_motor0(jnp.clip(o[0], -1.0, 1.0), i) for i, o in enumerate(outs)]
        if "expert" in params:  # 5-mode policy (IDEA-016): the healthy expert is mode 4 and is trained too
            act_e, value_e = expert_out(params, obs)
            outs.append((None, None, value_e))
            acts.append(act_e)
        w = mix_weights(params, obs)
        wg = jax.lax.stop_gradient(w)
        means = jnp.stack(acts, 1)
        value = jnp.einsum("ni,ni->n", wg, jnp.stack([o[2] for o in outs], -1))
        mean = jnp.einsum("ni,nia->na", wg, means)
        return mean.reshape(lead + (-1,)), outs[0][1], value.reshape(lead), w.reshape(lead + (-1,))


def fault_label(e, modes=4):
    """Selector target (privileged, training only): failed motor once its fault has acted, else uniform over the
    4 copies, or 'healthy' (mode 4) when a healthy expert exists."""
    faulted = (e.t > e.t_fail) & (jnp.min(e.k, -1) < 0.5)
    healthy = jnp.full(4, 0.25) if modes == 4 else jax.nn.one_hot(4, 5)
    return jnp.where(faulted[:, None], jax.nn.one_hot(jnp.argmin(e.k, -1), modes), healthy)


def train_selector(a, out, log):
    """IDEA-008: supervised selector training on rollouts of the current mixture policy (DAgger style),
    loss ||w - label||^2 (paper eq. 4). The motor-0 base policy stays frozen."""
    n, T = a.envs, a.horizon
    sim = ftc.make_sim(n, gz=a.gz_rand)
    sim_step = sim.build_step_fn()
    ep_len = int(ftc.T_END * ftc.CTRL_HZ)
    base = pickle.load(open(a.base, "rb"))
    frozen = {"base": base} | ({"expert": pickle.load(open(a.expert, "rb"))} if a.expert else {})
    modes = n_modes(frozen)
    tx = optax.adam(a.lr)

    def env_step(carry, key):
        sel, e, obs = carry
        k_act, k_reset = jax.random.split(key)
        mean, _ = symmetric_mean(frozen | {"selector": sel}, obs)
        action = jnp.clip(mean + a.act_noise * jax.random.normal(k_act, mean.shape), -1.0, 1.0)
        label = fault_label(e, modes)
        e, _, crashed = ftc.step(sim_step, e, action)
        done = crashed | (e.t >= ep_len)
        e = ftc.reset_where(e, done, *sample_resets(k_reset, n, a))
        return (sel, e, ftc.observe(e)), (obs, label, done, crashed)

    def loss_fn(sel, obs, label):
        return jnp.sum((Selector(modes).apply(sel, obs) - label) ** 2, -1).mean()

    @jax.jit
    def iteration(sel, opt_state, e, obs, key):
        key, k_roll, k_perm = jax.random.split(key, 3)
        (sel, e, obs), (o, label, done, crashed) = jax.lax.scan(env_step, (sel, e, obs), jax.random.split(k_roll, T))
        o, label = o.reshape(T * n, -1), label.reshape(T * n, -1)
        mb = T * n // a.minibatches
        perm = jax.random.permutation(k_perm, T * n)[: mb * a.minibatches].reshape(a.minibatches, mb)

        def minibatch(carry, idx):
            sel, opt_state = carry
            updates, opt_state = tx.update(jax.grad(loss_fn)(sel, o[idx], label[idx]), opt_state)
            return (optax.apply_updates(sel, updates), opt_state), None

        for _ in range(a.epochs):
            (sel, opt_state), _ = jax.lax.scan(minibatch, (sel, opt_state), perm)
        w = Selector(modes).apply(sel, o)
        faulted = label[:, :4].max(-1) > 0.5
        acc = jnp.sum(faulted * (w.argmax(-1) == label.argmax(-1))) / jnp.maximum(faulted.sum(), 1)
        return sel, opt_state, e, obs, key, jnp.array([loss_fn(sel, o, label), done.sum(), crashed.sum(), acc])

    key = jax.random.key(a.seed)
    k_net, k_reset, key = jax.random.split(key, 3)
    e = ftc.reset_where(ftc.init(sim), jnp.ones(n, bool), *sample_resets(k_reset, n, a))
    obs = ftc.observe(e)
    sel = Selector(modes).init(k_net, obs[:1])
    opt_state = tx.init(sel)
    t0, i = time.time(), 0
    while i * n * T < a.steps:
        sel, opt_state, e, obs, key, stats = iteration(sel, opt_state, e, obs, key)
        i += 1
        if i % 10 == 0 or (i + 1) * n * T >= a.steps:
            loss, dones, crashes, acc = np.asarray(stats)
            line = (f"{i * n * T / 1e6:8.2f}M steps  {i * n * T / (time.time() - t0):9.0f} steps/s  sel loss {loss:.3f}"
                    f"  episodes {int(dones):5d}  crash {crashes / max(dones, 1):.2f}  selector acc {acc:.2f}")
            print(line, flush=True)
            print(line, file=log, flush=True)
            pickle.dump(jax.device_get(frozen | {"selector": sel}), open(out / "policy.pkl", "wb"))


def sample_resets(key, n, a):
    """Random start in a 1 m cube around the target (paper), random failed motor, fault time and severity."""
    kp, km, kt, kf, ks, kk = jax.random.split(key, 6)
    pos = ftc.TARGET + jax.random.uniform(kp, (n, 3), minval=-0.5, maxval=0.5)
    motor = jnp.full(n, a.motor) if a.motor >= 0 else jax.random.randint(km, (n,), 0, 4)
    t_fail = (jax.random.uniform(kt, (n,), minval=a.t_fail[0], maxval=a.t_fail[1]) * ftc.CTRL_HZ).astype(jnp.int32)
    severity = jnp.where(jax.random.uniform(ks, (n,)) < a.partial, jax.random.uniform(kk, (n,), minval=a.severity[0], maxval=a.severity[1]), 0.0)
    health = jnp.where(jax.random.uniform(kf, (n,)) < a.p_fault, severity, 1.0)
    k = jnp.where(jax.nn.one_hot(motor, 4) > 0, health[:, None], 1.0)
    rand = None
    if a.gz_rand:  # Gazebo-gap randomisation, drawn per episode (ranges from the PX4/Gazebo bench report)
        kd, kc, kl, kn, kj, ksp, ka, kr, ke = jax.random.split(jax.random.fold_in(key, 1), 9)
        rand = {"delay": jax.random.randint(kd, (n,), a.delay_min, a.delay_max + 1),  # actuation delay, 2 ms steps
                "drag_c": jax.random.uniform(kc, (n,), minval=0.5, maxval=1.5),  # x Gazebo rotor drag
                "lag": jax.random.randint(kl, (n,), 0, a.lag_max + 1),  # estimate lag, 10 ms steps
                "noise": jax.random.uniform(kn, (n,)),  # 0-1 x EST_SIGMA
                "floor": ftc.F_FLOOR}
        if a.prop_j:  # rotor inertia, log-uniform: Gazebo's 0.1x up to the physical rotor
            rand["prop_j"] = jnp.exp(jax.random.uniform(kj, (n,), minval=np.log(a.prop_j[0]), maxval=np.log(a.prop_j[1])))
        rand["spin"] = (jax.random.uniform(ksp, (n,)) < a.p_spin).astype(jnp.float32)  # prop loss instead of a stop
        if a.moving:  # a share of episodes starts in flight (bench phases: forward 2 m/s, climb/descent 1 m/s, yaw 1.5
            # rad/s) with any heading, and faults 0-0.5 s later, before the controller has slowed it down
            km, kv, kz, kw, kt2 = jax.random.split(jax.random.fold_in(key, 2), 5)
            mv = jax.random.uniform(km, (n,)) < a.moving
            ang = jax.random.uniform(kv, (n,), maxval=2 * np.pi)
            sp = jax.random.uniform(jax.random.fold_in(kv, 1), (n,), maxval=2.5)
            v = jnp.stack([sp * jnp.cos(ang), sp * jnp.sin(ang), jax.random.uniform(kz, (n,), minval=-1.2, maxval=1.2)], -1)
            rand["vel"] = v * mv[:, None]
            rand["wz"] = mv * jax.random.uniform(kw, (n,), minval=-2.0, maxval=2.0)
            t_fail = jnp.where(mv, (jax.random.uniform(kt2, (n,), maxval=0.5) * ftc.CTRL_HZ).astype(jnp.int32), t_fail)
            rand["yaw"] = jax.random.uniform(jax.random.fold_in(key, 3), (n,), minval=-np.pi, maxval=np.pi)  # any heading
        if a.ekf_rand:  # EKF attitude error after the fault in a share of episodes: up to ekf_rand deg at 10-60 deg/s
            on = jax.random.uniform(ke, (n,)) < 0.7
            rand["att_max"] = on * jax.random.uniform(ka, (n,), maxval=np.radians(a.ekf_rand))
            rand["att_rate"] = jax.random.uniform(kr, (n,), minval=np.radians(10), maxval=np.radians(60))
        if a.hw_rand:  # plant deviations a real airframe has (ftc.NOMINAL "hardware" keys); bench_sim hwx_* tests wider
            kh = jax.random.split(jax.random.fold_in(key, 4), 7)
            ang = jax.random.uniform(kh[1], (n,), maxval=2 * np.pi)
            r = jax.random.uniform(kh[2], (n,), maxval=0.02)
            # the hard case (i023b crashed only there): CG offset toward the failed rotor, which no thrust can balance
            to_failed = ftc.ROTOR_POS[motor, :2] / jnp.linalg.norm(ftc.ROTOR_POS[motor, :2], axis=-1, keepdims=True)
            hard = jax.random.uniform(jax.random.fold_in(kh[2], 1), (n, 1)) < a.cg_toward
            toward = jnp.pad(hard * to_failed * jax.random.uniform(jax.random.fold_in(kh[2], 2), (n, 1), maxval=0.02), ((0, 0), (0, 1)))
            r = jnp.where(hard[:, 0], r * 0.5, r)
            rand |= {"mass": jax.random.uniform(kh[0], (n,), minval=0.9, maxval=1.1),
                     "cg": jnp.stack([r * jnp.cos(ang), r * jnp.sin(ang), jnp.zeros(n)], -1) + toward,
                     "t_err": jax.random.uniform(kh[3], (n, 4), minval=0.9, maxval=1.1),
                     "sag": jax.random.uniform(kh[4], (n,), minval=0.85, maxval=1.0),
                     "motor_tau": jax.random.uniform(kh[5], (n,), minval=0.8, maxval=a.motor_tau_max),
                     "J": jax.random.uniform(kh[6], (n,), minval=0.8, maxval=1.2)}
        if a.tilt_rand:  # bent / misaligned motor mounts in 70 % of episodes: each rotor 0-tilt_rand deg, any direction
            kt3 = jax.random.split(jax.random.fold_in(key, 6), 3)
            ta = jax.random.uniform(kt3[0], (n, 4), maxval=2 * np.pi)
            tm = jax.random.uniform(kt3[1], (n, 4), maxval=np.radians(a.tilt_rand)) * (jax.random.uniform(kt3[2], (n, 1)) < 0.7)
            rand["tilt"] = jnp.stack([jnp.cos(ta), jnp.sin(ta), jnp.zeros_like(ta)], -1) * tm[..., None]
        if a.onboard:  # bench onboard estimator (gyro attitude + GPS KF) in a share P of episodes
            p_on, s_xy, s_z, bias, head = a.onboard
            ko = jax.random.split(jax.random.fold_in(key, 5), 9)
            unit = lambda k: (lambda d: d / jnp.linalg.norm(d, axis=-1, keepdims=True))(jax.random.normal(k, (n, 3)))  # noqa: E731
            rand |= {"kf": (jax.random.uniform(ko[0], (n,)) < p_on).astype(jnp.float32),
                     "g_s": jax.random.uniform(ko[1], (n, 3), minval=-1.0, maxval=1.0) * jnp.array([s_xy, s_xy, s_z]) / 100,
                     "g_b": unit(ko[2]) * jax.random.uniform(ko[3], (n, 1), maxval=np.radians(bias)),
                     "head": jax.random.uniform(ko[4], (n,), minval=-1.0, maxval=1.0) * np.radians(head),
                     "att_seed": unit(ko[5]) * jax.random.uniform(ko[6], (n, 1), maxval=np.radians(1.0)),
                     "gps_period": jax.random.randint(ko[7], (n,), 10, 21), "gps_delay": jax.random.randint(ko[8], (n,), 10, 21),
                     "gps_vn": jax.random.uniform(jax.random.fold_in(ko[0], 1), (n,), minval=0.1, maxval=0.3),
                     "gps_pj": jax.random.uniform(jax.random.fold_in(ko[0], 2), (n,), maxval=0.05)}
    return pos, t_fail, k, rand


def make_train(a):
    n, T = a.envs, a.horizon
    sim = ftc.make_sim(n, gz=a.gz_rand)
    sim_step = sim.build_step_fn()
    ep_len = int(ftc.T_END * ftc.CTRL_HZ)
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr))
    NET = Joint() if a.joint else ActorCritic(selector=a.selector)
    modes = n_modes(pickle.load(open(a.init, "rb"))) if a.joint else 4
    expert = pickle.load(open(a.expert, "rb")) if a.expert else None

    def env_step(carry, key):
        params, e, obs = carry
        k_act, k_reset = jax.random.split(key)
        mean, log_std, value, _ = NET.apply(params, obs)
        # selector label: failed motor once its fault has acted (privileged, training only), else uniform
        label = fault_label(e, modes)
        action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
        logp = jnp.sum(-0.5 * ((action - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), -1)
        env_action, learner = jnp.clip(action, -1.0, 1.0), jnp.ones(n, bool)
        if expert is not None:  # IDEA-013: the healthy expert flies until the handoff; those steps are not trained on
            # shortcut: 0-300 ms handoff delay drawn from t_fail by a hash, not a stored per-episode draw
            learner = e.t >= e.t_fail + (e.t_fail * 7919) % 31
            env_action = jnp.where(learner[:, None], env_action, policy_mean(expert, obs))
        if a.handover:  # PX4 flies (logged reaction, ftc.px4_standin) from the fault to the handover; not trained on
            # shortcut: 0-handover ms drawn from t_fail by a hash, not a stored per-episode draw
            px4 = (e.t >= e.t_fail) & (jnp.min(e.k, -1) < 0.5) & (e.t < e.t_fail + (e.t_fail * 7919) % (a.handover // 10 + 1))
            learner = learner & ~px4
            env_action = jnp.where(px4[:, None], ftc.px4_standin(e), env_action)
        e, reward, crashed = ftc.step(sim_step, e, env_action)
        done = crashed | (e.t >= ep_len)
        e = ftc.reset_where(e, done, *sample_resets(k_reset, n, a))
        tr = (obs, action, logp, value, reward * a.reward_scale, done, crashed, label, learner)
        return (params, e, ftc.observe(e)), tr

    def loss_fn(params, obs, action, logp_old, adv, ret, label, m):
        mean, log_std, value, w = NET.apply(params, obs)
        logp = jnp.sum(-0.5 * ((action - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), -1)
        ratio = jnp.exp(logp - logp_old)
        m = m.astype(jnp.float32)  # 0 on steps the learner did not act (IDEA-013 expert phase), else 1
        avg = lambda x: jnp.sum(m * x) / jnp.maximum(m.sum(), 1.0)  # noqa: E731
        adv = (adv - avg(adv)) / (jnp.sqrt(avg((adv - avg(adv)) ** 2)) + 1e-8)
        pg = -avg(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * adv))
        vf = 0.5 * avg((value - ret) ** 2)
        ent = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
        sel = jnp.sum((w - label) ** 2, -1).mean() if a.selector or a.joint else 0.0  # paper eq. 4
        return pg + a.vf_coef * vf - a.ent_coef * ent + a.sel_coef * sel

    @jax.jit
    def iteration(params, opt_state, e, obs, key):
        key, k_roll, k_perm = jax.random.split(key, 3)
        (params, e, obs), (o, act, logp, val, rew, done, crashed, label, learner) = jax.lax.scan(
            env_step, (params, e, obs), jax.random.split(k_roll, T))
        last_val = NET.apply(params, obs)[2]
        w = NET.apply(params, o)[3]
        faulted = label[..., :4].max(-1) > 0.5
        sel_acc = jnp.sum(faulted * (w.argmax(-1) == label.argmax(-1))) / jnp.maximum(faulted.sum(), 1)

        def gae(carry, x):  # truncation at 15 s is treated as termination (small bias, simpler)
            adv_next, v_next = carry
            r, d, v = x
            delta = r + a.gamma * v_next * (1 - d) - v
            adv = delta + a.gamma * a.lam * (1 - d) * adv_next
            return (adv, v), adv

        _, adv = jax.lax.scan(gae, (jnp.zeros(n), last_val), (rew, done, val), reverse=True)
        ret = adv + val
        flat = lambda x: x.reshape((T * n,) + x.shape[2:])
        batch = tuple(map(flat, (o, act, logp, adv, ret, label, learner)))
        mb = T * n // a.minibatches

        def epoch(carry, k):
            params, opt_state = carry
            perm = jax.random.permutation(k, T * n)[: mb * a.minibatches].reshape(a.minibatches, mb)

            def minibatch(carry, idx):
                params, opt_state = carry
                grads = jax.grad(loss_fn)(params, *(x[idx] for x in batch))
                updates, opt_state = tx.update(grads, opt_state)
                return (optax.apply_updates(params, updates), opt_state), None

            return jax.lax.scan(minibatch, (params, opt_state), perm)[0], None

        (params, opt_state), _ = jax.lax.scan(epoch, (params, opt_state), jax.random.split(k_perm, a.epochs))
        log_std = (params["base"] if a.joint else params)["params"]["log_std"]
        stats = jnp.array([rew.mean() / a.reward_scale, done.sum(), crashed.sum(), jnp.exp(log_std).mean(), sel_acc])
        return params, opt_state, e, obs, key, stats

    def init(key):
        k_net, k_reset, key = jax.random.split(key, 3)
        e = ftc.reset_where(ftc.init(sim), jnp.ones(n, bool), *sample_resets(k_reset, n, a))
        obs = ftc.observe(e)
        params = pickle.load(open(a.init, "rb")) if a.init else NET.init(k_net, obs[:1])
        if a.equivariant and "selector" in params:  # warm start: same selector weights, used per motor frame
            params["selector_eq"] = params.pop("selector")
        if a.reset_std:  # restart exploration from a collapsed std
            p = params["base"] if a.joint else params
            p["params"]["log_std"] = jnp.full_like(p["params"]["log_std"], np.log(a.reset_std))
        return params, tx.init(params), e, obs, key

    return init, iteration


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--steps", type=float, default=5e6)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--t-fail", type=float, nargs=2, default=[1.0, 4.0], help="fault time range, s")
    ap.add_argument("--motor", type=int, default=-1, help="-1 = random motor")
    ap.add_argument("--p-fault", type=float, default=0.9, help="share of episodes with a fault")
    ap.add_argument("--partial", type=float, default=0.0, help="share of faults with random severity")
    ap.add_argument("--equivariant", action="store_true", help="--joint 5-mode: exactly symmetric selector and expert")
    ap.add_argument("--severity", type=float, nargs=2, default=[0.0, 1.0], help="--partial: remaining thrust share range")
    ap.add_argument("--init", default=None, help="policy.pkl to continue from (ladder rungs)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--selector", action="store_true", help="paper's Selector-Controller actor")
    ap.add_argument("--sel-coef", type=float, default=1.0)
    ap.add_argument("--base", default=None, help="motor-0 policy.pkl: train only the symmetric selector (IDEA-008)")
    ap.add_argument("--act-noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--joint", action="store_true", help="IDEA-012: PPO on the symmetric mixture (--init base+selector)")
    ap.add_argument("--expert", default=None, help="healthy policy.pkl: flies before the handoff (IDEA-013), or mode 4 of the selector with --base")
    ap.add_argument("--reset-std", type=float, default=None, help="reset the policy std after --init")
    ap.add_argument("--gz-rand", action="store_true", help="randomise over the Gazebo gap (delay, rotor drag, estimate lag/noise, floor)")
    ap.add_argument("--reward", default="paper", choices=["paper", "smooth", "precise"], help="smooth = IDEA-009")
    ap.add_argument("--tilt-pen", type=float, default=0.0, help="reward penalty weight on tilt beyond 70 deg")
    ap.add_argument("--moving", type=float, default=0.0, help="--gz-rand: share of episodes that start in flight")
    ap.add_argument("--handover", type=int, default=0, help="max ms of PX4's reaction after the fault before the policy flies")
    ap.add_argument("--delay-max", type=int, default=ftc.DELAY_MAX, help="--gz-rand: longest actuation delay, 2 ms physics steps")
    ap.add_argument("--lag-max", type=int, default=2, help="--gz-rand: longest estimate lag, 10 ms policy steps")
    ap.add_argument("--motor-tau-max", type=float, default=2.0, help="--hw-rand: largest motor time constant x nominal")
    ap.add_argument("--tilt-rand", type=float, default=0.0, help="--gz-rand: max thrust-axis tilt per rotor, deg")
    ap.add_argument("--delay-min", type=int, default=2, help="--gz-rand: shortest actuation delay, 2 ms physics steps")
    ap.add_argument("--prop-j", type=float, nargs=2, default=None, help="--gz-rand: rotor inertia range x physical")
    ap.add_argument("--p-spin", type=float, default=0.0, help="--gz-rand: share of faults that are prop loss")
    ap.add_argument("--hw-rand", action="store_true", help="--gz-rand: randomise mass, CG, thrust curves, sag, motor lag, J")
    ap.add_argument("--cg-toward", type=float, default=0.0, help="--hw-rand: share of episodes with the CG 0-2 cm toward the failed rotor")
    ap.add_argument("--onboard", type=float, nargs=5, default=None, metavar=("P", "SCALE_XY", "SCALE_Z", "BIAS", "HEAD"),
                    help="--gz-rand: bench onboard estimator in share P: gyro scale %% (x/y, z), bias deg/s, heading deg")
    ap.add_argument("--snaps", action="store_true", help="also keep runs/<id>/snaps/<steps>.pkl at every log line (training_video.py)")
    ap.add_argument("--ekf-rand", type=float, default=0.0, help="--gz-rand: max EKF attitude error after the fault, deg")
    a = ap.parse_args()

    ftc.REWARD, ftc.TILT_PEN = a.reward, a.tilt_pen
    out = Path(__file__).parent / "runs" / a.id
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "w")
    print(vars(a), file=log, flush=True)
    if a.base:
        train_selector(a, out, log)
        raise SystemExit
    init, iteration = make_train(a)
    params, opt_state, e, obs, key = init(jax.random.key(a.seed))
    per_iter = a.envs * a.horizon
    t0, i = time.time(), 0
    while i * per_iter < a.steps:
        params, opt_state, e, obs, key, stats = iteration(params, opt_state, e, obs, key)
        i += 1
        if i % 10 == 0 or (i + 1) * per_iter >= a.steps:
            r, dones, crashes, std, sel_acc = np.asarray(stats)
            line = (f"{i * per_iter / 1e6:8.2f}M steps  {i * per_iter / (time.time() - t0):9.0f} steps/s  "
                    f"reward {r:.3f}  episodes {int(dones):5d}  crash {crashes / max(dones, 1):.2f}  std {std:.3f}"
                    + (f"  selector acc {sel_acc:.2f}" if a.selector or a.joint else ""))
            print(line, flush=True)
            print(line, file=log, flush=True)
            pickle.dump(jax.device_get(params), open(out / "policy.pkl", "wb"))
            if a.snaps:
                (out / "snaps").mkdir(exist_ok=True)
                pickle.dump(jax.device_get(params), open(out / "snaps" / f"{i * per_iter:012d}.pkl", "wb"))
    pickle.dump(jax.device_get(params), open(out / "policy.pkl", "wb"))
