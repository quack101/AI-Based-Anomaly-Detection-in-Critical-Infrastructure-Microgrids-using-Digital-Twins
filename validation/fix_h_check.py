"""
Branch fix-h-yload, A2/A3 -- checks for h_model="network_y"
(observability/estimator_model.py).

A2  H vs central finite differences of h, for both h models, at 10
    states (x_true of 10 baseline points, used here only as evaluation
    points), along 20 random directions each (scaled coordinates
    y = x / state_scale, eps 1e-6): per row, |H d - FD| / max(|FD|, 1e-9 *
    max|FD|); max per row class.
A3  Consistency at the true state, 90 baseline points, sigmas S1 (corpus
    noise sigmas, config/sigma_overrides_s1.json): e = (z - h(x_true)) /
    sigma for every weighted row; per class median |mean e| and median
    RMS, legacy vs network_y; ZI residuals |h_zi(x_true)|. Rows with
    |mean e| > 3 under network_y are listed with a structural check (other
    loads on the same bus/phase, regulator / capacitor / switch bus).
    For rows with a structural reason: sigma such that RMS e = 1 on the
    TRAIN split (corpus A train telemetry + train x_true, stride 200),
    written to config/sigma_overrides_structural.json. A3 is then
    re-reported with S1 + those overrides.

Ground truth is used only for these checks and for the train-split
sigma of structurally mismatched rows (the same train-only use REGZI_*
statistics already make). One process.

Run: python -m validation.fix_h_check
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability.pseudo_measurements import build_pseudo_entries_for_hour
from observability import jacobian as J
from observability.zero_injection import _regulator_buses, _capacitor_buses, _switch_buses
from estimation.state_estimator import StateEstimator
from validation.exp_p1_corpus import load_corpus, point_sets, CORPORA, S1_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "validation"
STRUCTURAL_PATH = BASE_DIR / "config" / "sigma_overrides_structural.json"
TRAIN_STRIDE = 200
N_FD_STATES = 10
N_FD_DIRS = 20
FD_EPS = 1e-6


def row_class(e):
    i, q = e["id"], e["quantity"]
    if i.startswith("PSEUDO_"):
        return "PSEUDO"
    if i.startswith("REGZI_"):
        return "REGZI"
    if i.startswith("ZI_"):
        return "ZI"
    if q == "P_head":
        return "HEAD_P"
    if q == "Q_head":
        return "HEAD_Q"
    if q == "V_magnitude":
        return "meter V"
    return {"P_injection": "meter P", "Q_injection": "meter Q"}[q]


def weighted_rows(est, model, ts):
    return est._weighted_entries_for_hour(int(ts[11:13])) + model.zi_entries


def h_at(model, entries, xv, xd):
    st = model.state
    return J.build_jacobian(entries, model.Y, model.node_index, st, at_v=complex_v_from_state(st, xv, xd))


def fd_check(model, est, df, x_v, x_d, points):
    st = model.state
    n = st.n_phase_nodes
    s = est._state_scale
    rng = np.random.default_rng(12345)
    worst = defaultdict(float)
    for i in points[:N_FD_STATES]:
        ts = str(df.iloc[i]["real_timestamp"])
        ents = weighted_rows(est, model, ts)
        cls = [row_class(e) for e in ents]
        H, _ = h_at(model, ents, x_v[i], x_d[i])
        Hs = H * s[None, :]
        for _ in range(N_FD_DIRS):
            d = rng.standard_normal(st.dim)
            d /= np.linalg.norm(d)
            dx = FD_EPS * s * d
            _, hp = h_at(model, ents, x_v[i] + dx[:n], x_d[i] + dx[n:])
            _, hm = h_at(model, ents, x_v[i] - dx[:n], x_d[i] - dx[n:])
            fd = (hp - hm) / (2 * FD_EPS)
            an = Hs @ d
            scale = np.maximum(np.abs(fd), 1e-9 * np.max(np.abs(fd)))
            rel = np.abs(an - fd) / scale
            for c, r in zip(cls, rel):
                worst[c] = max(worst[c], float(r))
    return dict(worst)


def bus_structure(model):
    """(bus, phase letter) -> number of Load elements on it; and the special bus sets."""
    loads_on = defaultdict(list)
    for name in dss.Loads.AllNames():
        dss.Circuit.SetActiveElement(f"Load.{name}")
        bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
        for ph in dss.CktElement.NodeOrder():
            if ph in (1, 2, 3):
                loads_on[(bus, "ABC"[ph - 1])].append(name)
    return loads_on, {b.lower() for b in _regulator_buses()}, {b.lower() for b in _capacitor_buses()}, \
        {b.lower() for b in _switch_buses()}


def consistency(model, est, df, z_cols, x_v, x_d, points, sigma_override):
    rows = defaultdict(list)
    zi = []
    meta = {}
    for i in points:
        row = df.iloc[i]
        ts = str(row["real_timestamp"])
        ents = est._weighted_entries_for_hour(int(ts[11:13]))
        _, h0 = h_at(model, ents + model.zi_entries, x_v[i], x_d[i])
        m = len(ents)
        zi.append(np.abs(h0[m:]))
        for e, h in zip(ents, h0[:m]):
            z = e["z_value"] if "z_value" in e else float(row[e["id"]])
            sig = sigma_override.get(e["id"], e["sigma"])
            rows[e["id"]].append((z - h) / sig)
            meta[e["id"]] = (row_class(e), e.get("node"), e.get("phase"), e.get("load_name"))
    per = pd.DataFrame([{"id": k, "class": meta[k][0], "node": meta[k][1], "phase": meta[k][2],
                         "load_name": meta[k][3], "mean_e": float(np.mean(v)),
                         "rms_e": float(np.sqrt(np.mean(np.square(v))))} for k, v in rows.items()])
    zi = np.concatenate(zi)
    return per, float(np.max(zi)), float(np.median(zi))


def class_table(per):
    return per.groupby("class").agg(n=("id", "size"), median_abs_mean_e=("mean_e", lambda x: float(x.abs().median())),
                                    median_rms_e=("rms_e", "median")).reset_index()


def main():
    with open(S1_PATH) as f:
        s1 = json.load(f)["sigma"]
    df, z_cols, x_v, x_d = load_corpus(CORPORA["P0"])
    _, base90 = point_sets(df, z_cols)

    results = {}
    for hm in ("legacy", "network_y"):
        model = build_estimator_model("multiphase", h_model=hm)
        est = StateEstimator(model)
        fd = fd_check(model, est, df, x_v, x_d, base90)
        per, zi_max, zi_med = consistency(model, est, df, z_cols, x_v, x_d, base90, s1)
        results[hm] = (model, est, fd, per, zi_max, zi_med)
        print(f"[{hm}] FD max rel error per class: { {k: f'{v:.2e}' for k, v in sorted(fd.items())} }", flush=True)
        print(f"[{hm}] ZI residual at x_true: max {zi_max:.3e} kW/kvar, median {zi_med:.3e}", flush=True)

    model, est, _, per_net, _, _ = results["network_y"]
    loads_on, reg_b, cap_b, sw_b = bus_structure(model)
    big = per_net[per_net["mean_e"].abs() > 3].copy()

    def reason(r):
        bus = str(r["node"]).lower()
        why = []
        if r["class"] in ("meter P", "meter Q"):
            others = [l for l in loads_on.get((bus, r["phase"]), []) if l.lower() != str(r["load_name"]).lower()]
            if others:
                why.append(f"bus/phase also carries {','.join(others)} (meter reads one load, h is the node total)")
        if bus in reg_b:
            why.append("regulator bus")
        if bus in cap_b:
            why.append("capacitor bus")
        if bus in sw_b:
            why.append("switch bus")
        return "; ".join(why) if why else "unknown"

    big["reason"] = big.apply(reason, axis=1)
    big["structural"] = big["reason"] != "unknown"
    print("\nrows with |mean e| > 3 under network_y (S1 sigmas):")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(big.sort_values("mean_e", key=abs, ascending=False).to_string(index=False))

    # structural overrides: sigma with RMS e = 1 on the TRAIN split (meter rows only -- the registry rows)
    struct_ids = big[big["structural"] & big["class"].isin(["meter P", "meter Q", "meter V", "HEAD_Q"])]["id"].tolist()
    tr_cols = pd.read_csv(CORPORA["P0"] / "train_telemetry.csv", nrows=0).columns.tolist()
    tr = pd.read_csv(CORPORA["P0"] / "train_telemetry.csv", usecols=["real_timestamp"] + struct_ids)
    tr_npz = np.load(CORPORA["P0"] / "train_x_true.npz", allow_pickle=True)
    tr_idx = list(range(0, len(tr), TRAIN_STRIDE))
    ents = [e for e in model.meas_entries if e["id"] in struct_ids]
    resid = defaultdict(list)
    for i in tr_idx:
        _, h0 = h_at(model, ents, tr_npz["x_v"][i], tr_npz["x_delta"][i])
        for e, h in zip(ents, h0):
            resid[e["id"]].append(float(tr.iloc[i][e["id"]]) - h)
    structural = {}
    for e in ents:
        r = np.array(resid[e["id"]])
        structural[e["id"]] = {"sigma": float(np.sqrt(np.mean(r ** 2))), "sigma_s1": s1.get(e["id"], e["sigma"]),
                               "train_mean_resid": float(np.mean(r)), "n_train": len(r),
                               "reason": big.loc[big["id"] == e["id"], "reason"].iloc[0]}
    with open(STRUCTURAL_PATH, "w") as f:
        json.dump({"source": "validation/fix_h_check.py: rows with |mean e| > 3 at x_true under h_model=network_y "
                             "with S1 sigmas and a known structural reason; sigma = RMS(z - h(x_true)) on the "
                             f"corpus A TRAIN split, every {TRAIN_STRIDE}th row ({len(tr_idx)} rows). Applied on "
                             "top of S1.",
                   "sigma": {k: v["sigma"] for k, v in structural.items()},
                   "detail": structural}, f, indent=1)
    print(f"\nstructural overrides ({len(structural)}):")
    for k, v in structural.items():
        print(f"  {k}: sigma {v['sigma_s1']:.4g} -> {v['sigma']:.4g}  (train mean resid {v['train_mean_resid']:.3f}; {v['reason']})")

    s1_struct = dict(s1, **{k: v["sigma"] for k, v in structural.items()})
    per_net2, zi_max2, _ = consistency(model, est, df, z_cols, x_v, x_d, base90, s1_struct)

    out = []
    for name, per in [("legacy S1", results["legacy"][3]), ("network_y S1", per_net),
                      ("network_y S1+structural", per_net2)]:
        t = class_table(per)
        t.insert(0, "setting", name)
        out.append(t)
    tab = pd.concat(out)
    tab.to_csv(OUT / "fix_h_consistency_by_class.csv", index=False)
    for name, per in [("legacy", results["legacy"][3]), ("network_y", per_net), ("network_y_struct", per_net2)]:
        per.assign(setting=name).to_csv(OUT / f"fix_h_consistency_rows_{name}.csv", index=False)
    pd.DataFrame([{"h_model": hm, "class": c, "max_rel_err": v} for hm in results for c, v in results[hm][2].items()]
                 ).to_csv(OUT / "fix_h_fd_check.csv", index=False)
    big.to_csv(OUT / "fix_h_rows_over3.csv", index=False)
    print()
    print(tab.to_string(index=False))
    still = per_net2[per_net2["mean_e"].abs() > 3]
    print(f"\nafter structural overrides, rows with |mean e| > 3: {len(still)}  {still['id'].tolist()}")
    print(f"ZI residual at x_true under network_y: max {zi_max2:.3e}")


if __name__ == "__main__":
    main()
