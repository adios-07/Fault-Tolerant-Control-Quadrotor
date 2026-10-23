# x500v2_i025a (passive fault-tolerant low-level controller)

Drop-in replacement for x500v2_i023b: same 27-in / 4-out policy.onnx, same pftc_ref.py, same node wiring.

1. In this folder: `sha256sum -c SHA256SUMS && python3 selftest.py` (needs numpy, onnxruntime), which must print PASS.
2. Point the pftc node's `policy` parameter at this folder's `policy.onnx` (or copy it over the old one). Nothing
   else changes: one PFTC object from take-off, `track()` with PX4's applied thrusts before detection, `step()` from
   the detection tick on, never reset at the handover. Thrust floor 0.27 N.
3. Rerun the GT, onboard-estimator and phase sets, and add partial loss (rotor at 30 / 50 / 65 / 80 % thrust; hand
   over on detection, or let the policy fly from take-off when the detector does not fire on mild faults).

What to expect (results.txt): every motor gives the same result (the policy is exactly symmetric); 0 crashes for
complete stop, prop loss and partial loss 30-80 % in hover and all phases; xy 0.07-0.13 m except partial 65-80 %
(0.4-0.7 m). Same limits as i023b: heading error must stay below about 20 deg (needs heading aiding), and the CG
within about 0.5 cm of the motor centre.
