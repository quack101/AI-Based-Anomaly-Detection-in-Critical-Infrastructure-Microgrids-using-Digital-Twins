"""
Branch exp-pseudo, A1 -- facts about the 42 unmetered loads (corpus A, P0).

Per load: OpenDSS load model, connection, kV, the pseudo P/Q table it is
estimated with (observability/pseudo_measurements.py), and on the 90
baseline points (same indices as validation/exp_p1_corpus.py):
  ratio = drawn P / pseudo P, where drawn P = the PSEUDO row's h(x_true)
          (the analytic injection at that bus/phase at the true state;
          ground truth used ONLY for this score),
  |V|^2 at the load's bus/phase from x_true (pu).
Whether the ratio tracks |V|^2: Pearson/Spearman across loads of the
per-load medians, and per load-model group.

Run: python -m validation.exp_pseudo_facts
"""

from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd
from scipy import stats as sstats

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability.pseudo_measurements import build_pseudo_entries_for_hour
from observability import jacobian as J
from validation.exp_p1_corpus import point_sets, load_corpus, CORPORA

OUT = Path(__file__).resolve().parent / "exp_pseudo_facts.csv"


def main():
    model = build_estimator_model("multiphase")
    state = model.state
    facts = {}
    for name in model.bus_phase_by_load:
        dss.Loads.Name(name)
        facts[name] = {"load": name, "model": dss.Loads.Model(), "is_delta": bool(dss.Loads.IsDelta()),
                       "kV": dss.Loads.kV(), "nominal_kW": dss.Loads.kW()}
        dss.Circuit.SetActiveElement(f"Load.{name}")
        facts[name]["phases"] = dss.CktElement.NumPhases()
        bus, ph = model.bus_phase_by_load[name]
        facts[name].update({"bus": bus, "phase": ph})

    df, z_cols, x_v, x_d = load_corpus(CORPORA["P0"])
    _, base90 = point_sets(df, z_cols)
    ratios, v2 = {n: [] for n in facts}, {n: [] for n in facts}
    for i in base90:
        ts = str(df.iloc[i]["real_timestamp"])
        entries = [e for e in build_pseudo_entries_for_hour(model.bus_phase_by_load, model.per_hour_by_load,
                                                            int(ts[11:13])) if e["quantity"] == "P_injection"]
        at_v = complex_v_from_state(state, x_v[i], x_d[i])
        _, h0 = J.build_jacobian(entries, model.Y, model.node_index, state, at_v=at_v)
        for e, h in zip(entries, h0):
            node = f"{e['node']}.{ {'A': 1, 'B': 2, 'C': 3}[e['phase']] }"
            vpu = x_v[i][state.v_index[node]] / state.v_base[node]
            ratios[e["load_name"]].append(h / e["z_value"])
            v2[e["load_name"]].append(vpu ** 2)

    rows = []
    for n, f in facts.items():
        rows.append({**f, "median_ratio_drawn_over_pseudo": float(np.median(ratios[n])),
                     "median_V2_pu": float(np.median(v2[n])),
                     "median_ratio_over_V2": float(np.median(np.array(ratios[n]) / np.array(v2[n])))})
    out = pd.DataFrame(rows).sort_values("load")
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    print()
    print("load model counts:", out["model"].value_counts().to_dict(), " delta:", int(out["is_delta"].sum()),
          " kV values:", out["kV"].round(3).value_counts().to_dict())
    r, v = out["median_ratio_drawn_over_pseudo"], out["median_V2_pu"]
    print(f"across loads: median ratio {r.median():.4f} (min {r.min():.4f}, max {r.max():.4f}); "
          f"median |V|^2 {v.median():.4f} (min {v.min():.4f}, max {v.max():.4f})")
    print(f"Pearson(ratio, |V|^2) = {sstats.pearsonr(r, v)[0]:.3f}; Spearman = {sstats.spearmanr(r, v)[0]:.3f}")
    for m, g in out.groupby("model"):
        print(f"  model {m}: n={len(g)} median ratio {g['median_ratio_drawn_over_pseudo'].median():.4f} "
              f"median |V|^2 {g['median_V2_pu'].median():.4f} median ratio/|V|^2 {g['median_ratio_over_V2'].median():.4f}")


if __name__ == "__main__":
    main()
