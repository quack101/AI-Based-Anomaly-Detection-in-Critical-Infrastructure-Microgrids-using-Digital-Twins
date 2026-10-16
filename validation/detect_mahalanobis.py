"""
Mahalanobis (whitened) aggregation of the v3 AE (d=32) per-channel
reconstruction errors, vs the max-aggregation. numpy only; no estimator
runs (the v3 AE per-channel errors are recomputed from its saved weights
-- an AE forward pass).

STEP 1  clean-TRAIN per-channel normalized errors e_t (168-dim); fit mu
        and Ledoit-Wolf shrunk covariance Sigma (shrinkage reported as
        lambda); report condition numbers before/after.
STEP 2  s_full = (e-mu)^T Sigma^-1 (e-mu); s_diag ignores correlations
        (diagonal of the sample covariance). Threshold each at the 99th
        percentile of clean VAL (1% val FAR).
STEP 3  score test once: realized FAR, attacked-step detection, window
        detection + family, per-timestep AUC, window AUC, localization
        (top Mahalanobis-contributing channel in the window's targets).
STEP 4  table vs BDD any-step and AE max-agg.

Run: python -m validation.detect_mahalanobis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_ae import forward, windows_of, W
from validation.detect_eval import roc_auc

OUT = Path(__file__).resolve().parent
FDIA = OUT.parent / "dataset" / "fdia_v1"
PARAM_KEYS = ["Wz", "Wr", "Wn", "Uz", "Ur", "Un", "bz", "br", "bn", "Wc", "bc", "Wd", "bd"]
L_FAR = 60


def ae_errors(P, mu_ae, sd_ae, r_c, X, batch=2048):
    S = (X - mu_ae) / sd_ae
    N, D = S.shape
    E = np.full((N, D), np.nan)
    idx = np.arange(W - 1, N)
    for b in range(0, len(idx), batch):
        ii = idx[b:b + batch]
        out, _ = forward(P, windows_of(S, ii))
        E[ii] = np.abs(X[ii] - (out * sd_ae + mu_ae)) / r_c
    return E


def ledoit_wolf(Xc):
    """Xc: (n,p) centered clean-train errors. Returns (Sigma_shrunk,
    sample_cov, shrink rho)."""
    n, p = Xc.shape
    S = (Xc.T @ Xc) / n
    m = np.trace(S) / p
    d2 = np.sum((S - m * np.eye(p)) ** 2) / p
    # b_bar^2 = mean_i ||x_i x_i^T - S||_F^2 / p
    xnorm4 = np.sum(Xc ** 2, axis=1) ** 2            # ||x_i||^4
    xtsx = np.einsum("ij,ij->i", Xc @ S, Xc)          # x_i^T S x_i
    bbar2 = (np.sum(xnorm4) - 2 * np.sum(xtsx) + n * np.sum(S ** 2)) / (n ** 2 * p)
    bbar2 = min(bbar2, d2)
    rho = bbar2 / d2 if d2 > 0 else 0.0
    Sigma = rho * m * np.eye(p) + (1 - rho) * S
    return Sigma, S, float(rho)


def segments(lab):
    att = lab["attacked"].to_numpy().astype(bool)
    n = len(lab)
    wins = [dict(wid=int(wid), rows=np.array(sorted(g.index)), type=g.iloc[0]["type"],
                 targets=g.iloc[0]["targets"]) for wid, g in lab[att].groupby("window_id")]
    blocks = []
    i = 0
    while i < n:
        if att[i]:
            i += 1; continue
        j = i
        while j < n and not att[j]:
            j += 1
        for b in range(i, j - L_FAR + 1, L_FAR):
            blocks.append(np.arange(b, b + L_FAR))
        i = j
    return wins, blocks


def evaluate(score, lab, thr, att):
    fin = np.isfinite(score)
    far = float(np.mean(score[fin & (att == 0)] > thr))
    step = float(np.mean(score[fin & (att == 1)] > thr))
    wins, blocks = segments(lab)
    det, fam = {}, {}
    for w in wins:
        rows = w["rows"][np.isfinite(score[w["rows"]])]
        dd = bool(np.any(score[rows] > thr)) if len(rows) else False
        det[w["wid"]] = dd
        fam.setdefault(w["type"], [0, 0]); fam[w["type"]][1] += 1; fam[w["type"]][0] += int(dd)
    def seg_max(rows):
        v = score[rows]; v = v[np.isfinite(v)]
        return float(v.max()) if len(v) else np.nan
    pos = np.array([seg_max(w["rows"]) for w in wins])
    neg = np.array([seg_max(b) for b in blocks])
    wauc = roc_auc(np.concatenate([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
    auc = roc_auc(score[fin], att[fin].astype(bool))
    return {"FAR": far, "step_det": step, "win_det": int(sum(det.values())), "n_win": len(wins),
            "by_family": {t: f"{a}/{n}" for t, (a, n) in fam.items()}, "per_ts_AUC": float(auc),
            "window_AUC": float(wauc), "det": det}





def main():
    global cols_global
    wz = np.load(OUT / "detect_ae_weights.npz", allow_pickle=True)
    P = {k: wz[k] for k in PARAM_KEYS}
    mu_ae, sd_ae, r_c = wz["mu"], wz["sd"], wz["r_c"]
    cols_global = [c for c in pd.read_csv(FDIA / "test_z.csv", nrows=0).columns if c != "timestamp"]

    Xtr = pd.read_csv(FDIA / "train_z.csv").drop(columns="timestamp").to_numpy(float)
    Xva = pd.read_csv(FDIA / "val_z.csv").drop(columns="timestamp").to_numpy(float)
    Xte = pd.read_csv(FDIA / "test_z.csv").drop(columns="timestamp").to_numpy(float)
    vlab = pd.read_csv(FDIA / "val_labels.csv"); tlab = pd.read_csv(FDIA / "test_labels.csv")

    Etr = ae_errors(P, mu_ae, sd_ae, r_c, Xtr)
    Eva = ae_errors(P, mu_ae, sd_ae, r_c, Xva)
    Ete = ae_errors(P, mu_ae, sd_ae, r_c, Xte)
    print("errors computed", flush=True)

    # STEP 1: fit on clean TRAIN (all train clean)
    tr = Etr[np.isfinite(Etr[:, 0])]
    mu = tr.mean(0)
    Xc = tr - mu
    Sigma, S, rho = ledoit_wolf(Xc)
    cond_S = float(np.linalg.cond(S)); cond_Sig = float(np.linalg.cond(Sigma))
    Minv = np.linalg.inv(Sigma)
    diag_var = np.diag(S) + 1e-9
    print(f"STEP1: Ledoit-Wolf shrinkage lambda(rho)={rho:.4f}  cond(sample)={cond_S:.3e} "
          f"cond(shrunk)={cond_Sig:.3e}  n_train={len(tr)}", flush=True)

    def maha_full(E):
        D = E - mu
        s = np.full(len(E), np.nan)
        fin = np.isfinite(E[:, 0])
        d = D[fin]
        s[fin] = np.einsum("ij,ij->i", d @ Minv, d)
        return s

    def maha_diag(E):
        D = E - mu
        s = np.full(len(E), np.nan)
        fin = np.isfinite(E[:, 0])
        s[fin] = np.sum(D[fin] ** 2 / diag_var, axis=1)
        return s

    att_v = vlab["attacked"].to_numpy(); att_t = tlab["attacked"].to_numpy()
    results = {}
    for name, fn, is_diag in [("maha_full", maha_full, False), ("maha_diag", maha_diag, True)]:
        sv, st = fn(Eva), fn(Ete)
        thr = float(np.quantile(sv[(att_v == 0) & np.isfinite(sv)], 0.99))
        ev = evaluate(st, tlab, thr, att_t)
        # localization
        hits = tot = 0
        wins, _ = segments(tlab)
        for w in wins:
            rows = w["rows"][np.isfinite(st[w["rows"]])]
            if not len(rows) or not np.any(st[rows] > thr):
                continue
            tot += 1
            contrib = np.zeros(Ete.shape[1])
            for rr in rows:
                dvec = Ete[rr] - mu
                c = dvec ** 2 / diag_var if is_diag else dvec * (Minv @ dvec)
                contrib += np.where(np.isfinite(c), c, 0.0)
            tgt = set(str(w["targets"]).split(";"))
            hits += int(cols_global[int(np.argmax(contrib))] in tgt)
        ev["localization"] = (hits / tot) if tot else float("nan")
        ev["loc_hits"], ev["loc_tot"] = hits, tot
        ev["threshold"] = thr
        results[name] = ev
        print(f"STEP3 {name}: FAR {ev['FAR']*100:.2f}% step {ev['step_det']*100:.2f}% "
              f"win {ev['win_det']}/{ev['n_win']} fam {ev['by_family']} AUC {ev['per_ts_AUC']:.4f} "
              f"winAUC {ev['window_AUC']:.4f} loc {hits}/{tot}", flush=True)

    summary = {"ledoit_wolf_lambda": rho, "cond_sample": cond_S, "cond_shrunk": cond_Sig,
               "n_train": int(len(tr)), "results": results,
               "baselines": {"bdd_anystep": "10/18 @ 1.13% FAR, per-ts AUC 0.607",
                             "ae_maxagg_v3": "7/18 @ 0.88% FAR, per-ts AUC 0.753"}}
    json.dump(summary, open(OUT / "detect_mahalanobis_summary.json", "w"), indent=2, default=float)
    rows = []
    for det, label, far, win, auc, wauc, fam, loc in [
        ("BDD any-step", "bdd", 0.0113, "10/18", 0.607, 0.784, {"A1": "8/12", "A2": "1/3", "A3": "1/3"}, None),
        ("AE max-agg (v3)", "ae", 0.0088, "7/18", 0.753, 0.806, {"A1": "5/12", "A2": "0/3", "A3": "2/3"}, None),
        ("AE Mahalanobis-full", "mf", results["maha_full"]["FAR"],
         f"{results['maha_full']['win_det']}/{results['maha_full']['n_win']}", results["maha_full"]["per_ts_AUC"],
         results["maha_full"]["window_AUC"], results["maha_full"]["by_family"], results["maha_full"]["localization"]),
        ("AE Mahalanobis-diag", "md", results["maha_diag"]["FAR"],
         f"{results['maha_diag']['win_det']}/{results['maha_diag']['n_win']}", results["maha_diag"]["per_ts_AUC"],
         results["maha_diag"]["window_AUC"], results["maha_diag"]["by_family"], results["maha_diag"]["localization"])]:
        rows.append({"detector": det, "realized_FAR": far, "window": win, "per_ts_AUC": auc,
                     "window_AUC": wauc, "A1": fam.get("A1"), "A2": fam.get("A2"), "A3": fam.get("A3"),
                     "localization": loc})
    pd.DataFrame(rows).to_csv(OUT / "detect_mahalanobis_table.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
