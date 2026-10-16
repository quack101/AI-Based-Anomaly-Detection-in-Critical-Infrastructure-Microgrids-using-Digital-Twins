"""
STEP A -- log per-channel WLS residual features from the FROZEN estimator
(estimator-v5-hfix; run, not changed). For each scored step: residual
vector r_c = (z_c - h_c(x_hat)) / sigma_c over ALL 361 weighted rows
(165 meter + 112 REGZI + 84 PSEUDO). No ground truth is read.

Strides: train every 4th clean step; val every 4th clean + every
attacked; test every step. Saved to residuals_{split}.npz (float32 R
matrix + ids + timestamps + attacked + converged) -- npz instead of a
130 MB CSV purely for size; the array is the per-channel residual log.
Checkpointed every 500 steps.

Run: python -m validation.detect_residual_log [--time50] [--splits a,b]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import StateEstimator

OUT = Path(__file__).resolve().parent
FDIA = OUT.parent / "dataset" / "fdia_v1"
CORPUS = OUT.parent / "simulation" / "phase6_corpus"
TRAIN_STRIDE = 4
CKPT = 500


def scored_rows(split, lab):
    att = lab["attacked"].to_numpy().astype(bool)
    n = len(lab)
    if split == "train":
        return list(range(0, n, TRAIN_STRIDE))                    # all clean
    if split == "val":
        clean = [i for i in range(0, n, TRAIN_STRIDE) if not att[i]]
        return sorted(set(clean) | set(np.flatnonzero(att).tolist()))
    return list(range(n))                                          # test: every step


def run_split(split, est, model, stride_train, out=True):
    state = model.state
    z = pd.read_csv(FDIA / f"{split}_z.csv")
    lab = pd.read_csv(FDIA / f"{split}_labels.csv")
    rts = pd.read_csv(CORPUS / f"{split}_telemetry.csv", usecols=["real_timestamp"])["real_timestamp"].tolist()
    z_cols = [c for c in z.columns if c != "timestamp"]
    rows = scored_rows(split, lab)
    we0 = est._weighted_entries_for_hour(0)
    ids = [e["id"] for e in we0]
    C = len(ids)
    R = np.full((len(rows), C), np.nan, dtype=np.float32)
    meta = []
    t0 = time.time()
    for n, i in enumerate(rows):
        zr = {c: float(z.iat[i, z.columns.get_loc(c)]) for c in z_cols}
        ts = str(rts[i])
        hour = int(ts[11:13])
        res = est.estimate(zr, ts)
        we = est._weighted_entries_for_hour(hour)
        zz = np.array([e["z_value"] if "z_value" in e else zr[e["id"]] for e in we])
        sig = np.array([e["sigma"] for e in we])
        at_v = complex_v_from_state(state, res.x_v, res.x_delta)
        _, h0 = J.build_jacobian(we, model.Y, model.node_index, state, at_v=at_v)
        R[n] = ((zz - h0) / sig).astype(np.float32)
        meta.append((int(z.iat[i, 0]), int(lab.iloc[i]["attacked"]), int(res.converged), i))
        if out and (n + 1) % CKPT == 0:
            _save(split, R[:n + 1], meta, ids)
            el = time.time() - t0
            print(f"  [{split}] {n+1}/{len(rows)} {el:.0f}s ({el/(n+1):.3f}s/row)", flush=True)
    if out:
        _save(split, R, meta, ids)
    conv = np.mean([m[2] for m in meta])
    print(f"[{split}] {len(rows)} rows, converged {conv*100:.2f}%, {time.time()-t0:.0f}s", flush=True)
    return len(rows), time.time() - t0


def _save(split, R, meta, ids):
    m = np.array(meta)
    np.savez(OUT / f"residuals_{split}.npz", R=R[:len(meta)], timestamp=m[:, 0], attacked=m[:, 1],
             converged=m[:, 2], row=m[:, 3], ids=np.array(ids, dtype=object))


def main():
    model = build_estimator_model("multiphase")
    est = StateEstimator(model)
    print(f"estimator m_weighted={est.m_weighted} dof={est.dof} h_model={model.h_model}", flush=True)

    if "--time50" in sys.argv:
        z = pd.read_csv(FDIA / "test_z.csv"); lab = pd.read_csv(FDIA / "test_labels.csv")
        rts = pd.read_csv(CORPUS / "test_telemetry.csv", usecols=["real_timestamp"])["real_timestamp"].tolist()
        z_cols = [c for c in z.columns if c != "timestamp"]
        t0 = time.time()
        for i in range(50):
            zr = {c: float(z.iat[i, z.columns.get_loc(c)]) for c in z_cols}
            est.estimate(zr, str(rts[i]))
        per = (time.time() - t0) / 50
        counts = {"train": len(range(0, 60480, TRAIN_STRIDE)), "test": 12960}
        vlab = pd.read_csv(FDIA / "val_labels.csv"); vatt = vlab["attacked"].to_numpy().astype(bool)
        counts["val"] = len(sorted(set([i for i in range(0, len(vlab), TRAIN_STRIDE) if not vatt[i]]) |
                                   set(np.flatnonzero(vatt).tolist())))
        print(f"per-step {per:.3f}s -> train {per*counts['train']/3600:.2f}h "
              f"val {per*counts['val']/3600:.2f}h test {per*counts['test']/3600:.2f}h "
              f"total {per*sum(counts.values())/3600:.2f}h", flush=True)
        return

    splits = ["train", "val", "test"]
    for a in sys.argv:
        if a.startswith("--splits"):
            splits = sys.argv[sys.argv.index(a) + 1].split(",")
    for sp in splits:
        run_split(sp, est, model, TRAIN_STRIDE)


if __name__ == "__main__":
    main()
