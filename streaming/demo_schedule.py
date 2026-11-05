"""
streaming/demo_schedule.py -- attack schedule of the full demo (preset C):
all five attack classes in one live run. Shared by streaming/run_demo.py
(writes the injector's attacks.json) and validation/gen_demo_attacks.py
(builds the A3/A4/A5 plans offline with OpenDSS).

Stream layout, relative to the start S (1 step = 1 simulated minute):
  [  0,  80) clean          (the V-peer check needs 30 steps of history)
  [ 80, 140) A1  20 random channels, P/Q +-10% of the reading, V +-0.01 pu
  [180, 240) A2  one meter's P, Q and V, same levels
  [280, 340) A3  load redistribution, 10% of total metered P (about 5% of the
                 feeder load) moved between two groups of 4 meters, every
                 reading made power-flow consistent
  [380, 410) A4  stealthy: state shift of 1e-4 pu in the estimator's kept
                 subspace (passes the residual test by construction)
  [450, 480) A5  stealthy, Newton-corrected A4 (nonlinear-consistent), 1e-4 pu
  [480, 520) clean
Start S = 80440 (preset C): a stretch of the test corpus where the clean
readings raise no AE / V-peer / P-peer alarm (start-selection check in
reports/demo_full.md). A3 at 10%: offline replay of this schedule with the
nine 5%-full bench A3 plans caught 5/9, with the 10% plans 9/9.
A4/A5 sizes: the library's 0.01 pu makes meters read MW (any detector sees
that); 1e-4 pu keeps the readings within tens of kW (reports/a3_a4_a5_detection.md).
"""

STEPS = 520
PLAN_DIR = "streaming/demo_attacks"


def schedule_C(S):
    return [
        {"type": "A1", "start": S + 80, "duration": 60, "profile": "step", "seed": 20261, "k": 20,
         "channel_class": "mixed", "pq_rel": 0.10, "v_abs_pu": 0.01},
        {"type": "A2", "start": S + 180, "duration": 60, "profile": "step", "seed": 20262,
         "pq_rel": 0.10, "v_abs_pu": 0.01},
        {"type": "A3", "start": S + 280, "duration": 60, "plan": f"{PLAN_DIR}/C_{S}_A3.npz", "magnitude": 0.10,
         "stealthy": True, "note": "zone load redistribution, full access",
         "build": {"profile": "step", "seed": 20263, "zone": True, "group_size": 4, "access": "full"}},
        {"type": "A4", "start": S + 380, "duration": 30, "plan": f"{PLAN_DIR}/C_{S}_A4.npz", "magnitude": 1e-4,
         "stealthy": True, "note": "kept-subspace stealthy attack",
         "build": {"profile": "step", "seed": 20264}},
        {"type": "A5", "start": S + 450, "duration": 30, "plan": f"{PLAN_DIR}/C_{S}_A5.npz", "magnitude": 1e-4,
         "stealthy": True, "note": "Newton-corrected stealthy attack",
         "build": {"profile": "step", "seed": 20265}},
    ]


def injector_specs(schedule):
    """attacks.json content: drop the offline build parameters."""
    return [{k: v for k, v in d.items() if k != "build"} for d in schedule]
