"""
Branch exp-p1 -- does the residual excess come from the 42 loads that
corpus A (P0) holds at their compiled nominal values? Compares P0
(simulation/phase6_corpus) with P1 (simulation/phase6_corpus_p1: same
generator, seed 42, meter set and corpus noise sigmas, all 91 loads
driven from dataset/node_loads_91).

STEP 2 -- for every PSEUDO_* row (84) at the 90 baseline points:
  e = (z_pseudo - h(x_true)) / sigma_pseudo, per corpus. x_true is used
  only for this score.
STEP 3 -- settings P0+S0, P0+S1, P1+S0, P1+S1 (S1 = config/
  sigma_overrides_s1.json at model build). Each setting's model takes
  its REGZI_* statistics from ITS corpus's train split
  (build_estimator_model train_x_true_path). Estimator unchanged
  (init="pseudo_pf" default, TSVD, HEAD_P handling).
  On the 60 scorecard solves and the 90 baseline points, using corpus
  A's point indices for BOTH corpora (paired by timestep).

One process (the power-flow start calls OpenDSS).

Run: python -m validation.exp_p1_corpus
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability.pseudo_measurements import build_pseudo_entries_for_hour
from observability import jacobian as J
from estimation.state_estimator import StateEstimator, TSVD_MAX_ITER
from estimation.bad_data_detection import compute_chi2_and_lnr
from attacks.a4_kept import kept_subspace_basis

BASE_DIR = Path(__file__).resolve().parent.parent
CORPORA = {"P0": BASE_DIR / "simulation" / "phase6_corpus", "P1": BASE_DIR / "simulation" / "phase6_corpus_p1"}
S1_PATH = BASE_DIR / "config" / "sigma_overrides_s1.json"
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"
OUT = BASE_DIR / "validation"
SEED = 777
N_PER_STRATUM = 10
N_EXTRA_CLEAN = 60


def stratified_picks(df, p_cols):
    total_p = df[p_cols].sum(axis=1)
    q1, q2 = total_p.quantile([1 / 3, 2 / 3])
    low = df.index[total_p <= q1]
    mid = df.index[(total_p > q1) & (total_p <= q2)]
    high = df.index[total_p > q2]
    rng = np.random.default_rng(SEED)

    def pick(idx, k):
        return sorted(rng.choice(idx, size=k, replace=False).tolist())

    return {"low": pick(low, N_PER_STRATUM), "mid": pick(mid, N_PER_STRATUM), "high": pick(high, N_PER_STRATUM)}


def load_corpus(cdir):
    cols = pd.read_csv(cdir / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(cdir / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols)
    npz = np.load(cdir / "test_x_true.npz", allow_pickle=True)
    return df, z_cols, npz["x_v"], npz["x_delta"]


def point_sets(df, z_cols):
    strata = stratified_picks(df.iloc[:1440], [c for c in z_cols if c.endswith("_P")])
    sweep30 = sorted(set(strata["low"] + strata["mid"] + strata["high"]))
    score60 = [j for idxs in strata.values() for i in idxs for j in (i, i + 1)]
    rng2 = np.random.default_rng(SEED + 1)
    remaining = [i for i in range(len(df)) if i not in sweep30]
    base90 = sweep30 + sorted(rng2.choice(remaining, size=N_EXTRA_CLEAN, replace=False).tolist())
    return score60, base90


def step2_pseudo(model, corpus, df, x_v, x_d, base90, metered):
    state = model.state
    rows = []
    for i in base90:
        ts = str(df.iloc[i]["real_timestamp"])
        entries = build_pseudo_entries_for_hour(model.bus_phase_by_load, model.per_hour_by_load, int(ts[11:13]))
        _, h0 = J.build_jacobian(entries, model.Y, model.node_index, state,
                                 at_v=complex_v_from_state(state, x_v[i], x_d[i]))
        for e, h in zip(entries, h0):
            rows.append({"corpus": corpus, "i": i, "id": e["id"], "load_name": e["load_name"],
                         "metered": e["load_name"] in metered, "e": (e["z_value"] - h) / e["sigma"]})
    return rows


def run_setting(name, corpus, ov, df, z_cols, x_v, score60, base90):
    model = build_estimator_model("multiphase", sigma_overrides=ov,
                                  train_x_true_path=CORPORA[corpus] / "train_x_true.npz")
    est = StateEstimator(model)
    state, dof = model.state, est.dof
    v_base = np.array([state.v_base[n] for n in state.all_node_names])
    lo95, hi95 = sstats.chi2.ppf(0.025, dof), sstats.chi2.ppf(0.975, dof)
    rows, summ = [], []
    for set_name, idxs in [("scorecard60", score60), ("baseline90", base90)]:
        t0 = time.time()
        errs, dom = [], Counter()
        sub = []
        for i in idxs:
            row = df.iloc[i]
            ts = str(row["real_timestamp"])
            z = {c: float(row[c]) for c in z_cols}
            res = est.estimate(z, ts)
            st = compute_chi2_and_lnr(est, res, z, ts)
            e = np.abs(res.x_v - x_v[i]) / v_base
            errs.append(e)
            nr = np.abs(np.asarray(st["normalized_residuals"], dtype=float))
            keep = np.isfinite(nr) & ~(np.asarray(st["redundancy"]) < 1e-3)
            dom[st["ids"][int(np.argmax(np.where(keep, nr, -1)))]] += 1
            sub.append({"setting": name, "set": set_name, "i": i, "converged": res.converged,
                        "iterations": res.iterations, "J": st["J"], "lnr": st["lnr"],
                        "constraint_residual_inf": res.constraint_residual_inf,
                        "kept": kept_subspace_basis(est, res, ts)["dim_K"],
                        "median_v_err_pu": float(np.median(e)), "max_v_err_pu": float(e.max())})
        sr = pd.DataFrame(sub)
        rows += sub
        pooled = np.concatenate(errs)
        node_max = np.max(np.vstack(errs), axis=0)
        summ.append({
            "setting": name, "set": set_name, "n": len(sr), "dof": dof,
            "converged": int(sr["converged"].sum()), "mean_iterations": sr["iterations"].mean(),
            "n_at_iter_cap": int((sr["iterations"] >= TSVD_MAX_ITER).sum()),
            "v_err_median": float(np.median(pooled)), "v_err_p95": float(np.percentile(pooled, 95)),
            "v_err_max": float(pooled.max()), "J_mean": sr["J"].mean(), "J_over_dof": sr["J"].mean() / dof,
            "n_in_chi2_95": int(((sr["J"] >= lo95) & (sr["J"] <= hi95)).sum()),
            "feasibility_max": sr["constraint_residual_inf"].max(),
            "kept_min": sr["kept"].min(), "kept_median": sr["kept"].median(), "kept_max": sr["kept"].max(),
            "worst3_nodes": "; ".join(f"{state.all_node_names[k]}={node_max[k]:.4f}"
                                      for k in np.argsort(-node_max)[:3]),
            "J99": sr["J"].quantile(0.99), "LNR99": sr["lnr"].quantile(0.99),
            "top5_lnr_channels": "; ".join(f"{c}={n}" for c, n in dom.most_common(5)),
            "runtime_s": time.time() - t0})
        print({k: summ[-1][k] for k in ("setting", "set", "converged", "mean_iterations", "v_err_median",
                                        "v_err_max", "J_over_dof", "n_in_chi2_95", "runtime_s")}, flush=True)
    return model, rows, summ


def main():
    with open(S1_PATH) as f:
        s1 = json.load(f)["sigma"]
    with open(REPRESENTATIVE_METERS) as f:
        metered = {m["load_name"] for m in json.load(f)}
    data = {c: load_corpus(d) for c, d in CORPORA.items()}
    # One point set for both corpora: corpus A's (the stratification uses the
    # telemetry P sum, which differs slightly in P1), so P0/P1 are paired by index.
    shared = point_sets(data["P0"][0], data["P0"][1])
    p1_own = point_sets(data["P1"][0], data["P1"][1])
    print(f"P1's own stratified sets would differ from P0's: scorecard {sorted(set(shared[0]) ^ set(p1_own[0]))[:6]}..., "
          f"baseline overlap {len(set(shared[1]) & set(p1_own[1]))}/90", flush=True)
    pts = {c: shared for c in CORPORA}

    all_rows, all_summ, pseudo = [], [], []
    for name, corpus, ov in [("P0+S0", "P0", None), ("P0+S1", "P0", s1), ("P1+S0", "P1", None), ("P1+S1", "P1", s1)]:
        df, z_cols, x_v, x_d = data[corpus]
        score60, base90 = pts[corpus]
        model, rows, summ = run_setting(name, corpus, ov, df, z_cols, x_v, score60, base90)
        all_rows += rows
        all_summ += summ
        if ov is None:  # pseudo values/sigmas do not depend on the meter sigmas
            pseudo += step2_pseudo(model, corpus, df, x_v, x_d, base90, metered)

    pd.DataFrame(all_rows).to_csv(OUT / "exp_p1_rows.csv", index=False)
    summary = pd.DataFrame(all_summ)
    summary.to_csv(OUT / "exp_p1_summary.csv", index=False)
    ps = pd.DataFrame(pseudo)
    per_row = ps.groupby(["corpus", "id", "load_name", "metered"])["e"].agg(
        mean="mean", rms=lambda x: float(np.sqrt(np.mean(np.square(x))))).reset_index()
    per_row.to_csv(OUT / "exp_p1_pseudo_rows.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    for c in ("P0", "P1"):
        pr = per_row[per_row["corpus"] == c]
        print(f"\n=== {c}: pseudo e over 84 rows: mean of |mean| {pr['mean'].abs().mean():.3f}, "
              f"median RMS {pr['rms'].median():.3f}, max RMS {pr['rms'].max():.3f}")
        print(pr.sort_values("rms", ascending=False).head(10).to_string(index=False))
    print()
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
