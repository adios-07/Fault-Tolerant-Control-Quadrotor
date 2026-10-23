#!/usr/bin/env python3
"""Export a symmetric Selector-Controller policy.pkl to ONNX: obs [N, 27] -> action [N, 4] in [-1, 1].
  python export_onnx.py runs/i014b/policy.pkl policies/x500v2_i014b/policy.onnx

The symmetry transforms are linear (sign flips plus a motor permutation), so copy i of the motor-0 base is
MatMul(obs, M_i) -> MLP -> Clip -> MatMul(., P_i). M_i and P_i are read off ftc itself by transforming identity
matrices, so the graph matches train.policy_mean by construction. A 5-mode policy adds the healthy expert.
After export, the ONNX output is checked against the JAX policy on observations from closed-loop rollouts.
"""
import pickle
import sys

import jax
import jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")  # GPU default is TF32: the JAX reference would be off by ~5e-3
import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper

import eval as ev
import ftc
import train


def build(params):
    nodes, inits = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, np.float32), name))
        return name

    def op(kind, ins, out, **kw):
        nodes.append(helper.make_node(kind, ins, [out], **kw))
        return out

    def weights(p, tag):
        return [(const(f"{tag}_W{j}", p["params"][f"Dense_{j}"]["kernel"]),
                 const(f"{tag}_b{j}", p["params"][f"Dense_{j}"]["bias"])) for j in range(4)]

    def mlp(x, ws, tag):  # 3x64 tanh + linear output, as train.mlp
        for j, (w, b) in enumerate(ws):
            x = op("Add", [op("MatMul", [x, w], f"{tag}_mm{j}"), b], f"{tag}_z{j}")
            if j < 3:
                x = op("Tanh", [x], f"{tag}_h{j}")
        return x

    lo, hi = const("lo", -1.0), const("hi", 1.0)
    base = weights(params["base"], "base")
    copies = []
    for i in range(4):
        M = np.asarray(ftc.obs_to_motor0(jnp.eye(ftc.OBS_DIM), i))  # obs @ M = obs_to_motor0(obs, i)
        P = np.asarray(ftc.act_from_motor0(jnp.eye(ftc.ACT_DIM), i))  # act @ P = act_from_motor0(act, i)
        a = op("Clip", [mlp(op("MatMul", ["obs", const(f"M{i}", M)], f"x{i}"), base, f"c{i}"), lo, hi], f"a{i}")
        copies.append(op("MatMul", [a, const(f"P{i}", P)], f"copy{i}"))
    eq = "selector_eq" in params  # exactly equivariant mixture (train.mix_weights / train.expert_out)
    if "expert" in params and eq:  # expert averaged over the 4 symmetries
        ew = weights(params["expert"], "expert")
        parts = [op("MatMul", [op("Clip", [mlp(f"x{i}", ew, f"e{i}"), lo, hi], f"ea{i}"), f"P{i}"], f"ep{i}") for i in range(4)]
        acc = parts[0]
        for i, part in enumerate(parts[1:], 1):
            acc = op("Add", [acc, part], f"esum{i}")
        copies.append(op("Mul", [acc, const("quarter", 0.25)], "copy4"))
    elif "expert" in params:
        copies.append(op("Clip", [mlp("obs", weights(params["expert"], "expert"), "e"), lo, hi], "copy4"))
    if eq:  # logit for "motor i failed" = selector (motor 0 - healthy) on x_i; healthy = 0
        sw, d = weights(params["selector_eq"], "sel"), const("d04", np.array([[1.0], [0.0], [0.0], [0.0], [-1.0]]))
        logits = [op("MatMul", [mlp(f"x{i}", sw, f"s{i}"), d], f"l{i}") for i in range(4)]
        logits.append(op("MatMul", ["obs", const("zcol", np.zeros((ftc.OBS_DIM, 1)))], "l4"))
        w = op("Softmax", [op("Concat", logits, "logits", axis=1)], "w", axis=-1)
    else:
        w = op("Softmax", [mlp("obs", weights(params["selector"], "sel"), "s")], "w", axis=-1)
    inits += [numpy_helper.from_array(np.array([d], np.int64), f"ax{d}") for d in (1, 2)]  # Unsqueeze/ReduceSum axes
    stacked = op("Concat", [op("Unsqueeze", [c, "ax1"], f"u{i}") for i, c in enumerate(copies)], "acts", axis=1)
    mixed = op("Mul", [stacked, op("Unsqueeze", [w, "ax2"], "w3")], "mixed")
    op("ReduceSum", [mixed, "ax1"], "action", keepdims=0)
    graph = helper.make_graph(
        nodes, "pftc", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", ftc.OBS_DIM])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["N", ftc.ACT_DIM])], inits)
    # IR 8 = what opset 17 needs; keeps the file loadable by older onnxruntime builds (e.g. on the ROS2 side)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    return model


def rollout_obs(params, fault):
    """Observations seen by the JAX policy on the 20 eval units (every 5th step)."""
    names, pos, k, t_fail = ev.units(fault)
    sim = ftc.make_sim(ev.N)
    sim_step = sim.build_step_fn()

    def ctrl(e, _):
        obs = ftc.observe(e)
        e, _, _ = ftc.step(sim_step, e, train.policy_mean(params, obs))
        return e, obs

    e = ftc.reset_where(ftc.init(sim), jnp.ones(ev.N, bool), pos, t_fail, k)
    _, obs = jax.jit(lambda e: jax.lax.scan(ctrl, e, length=int(ftc.T_END * ftc.CTRL_HZ)))(e)
    return np.asarray(obs[::5]).reshape(-1, ftc.OBS_DIM)


if __name__ == "__main__":
    pkl, out = sys.argv[1], sys.argv[2]
    params = pickle.load(open(pkl, "rb"))
    onnx.save(build(params), out)
    obs = np.concatenate([rollout_obs(params, True), rollout_obs(params, False)]).astype(np.float32)
    ref = np.asarray(train.policy_mean(params, jnp.asarray(obs)))
    got = onnxruntime.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"obs": obs})[0]
    err = float(np.abs(got - ref).max())
    print(f"{out}: {len(obs)} observations, max |onnx - jax| = {err:.2e}")
    assert err < 1e-5, err
