"""
streaming/detector_core.py -- the adopted detector (telemetry fusion:
physics BDD + AE-Mahalanobis) as a reusable scoring core for the live
Kafka path. It IMPORTS and LOADS the frozen offline artifacts; it never
retrains or reimplements the estimator, AE, Mahalanobis fit or fusion.

Frozen artifacts loaded (committed paths):
  - estimator  : estimator-v5-hfix = observability.estimator_model.
                 build_estimator_model("multiphase") defaults (network-Y
                 h_model, sigma overrides config/sigma_overrides_s1.json +
                 config/sigma_overrides_structural.json, pseudo_pf start)
                 + estimation.state_estimator.StateEstimator.
  - AE (d=32)  : validation/detect_ae_weights.npz (the v3 reconstruction
                 AE; forward pass via validation.detect_ae.forward).
  - BDD thresh : config/bdd_thresholds_val1pct.json (J99, LNR99).
  - fusion     : validation/detect_final_summary.json (logistic weights).
Derived-but-deterministic (recomputed once from the above + committed
data, exactly as validation/detect_final.py did, then cached to
streaming/detector_fit_cache.npz; NOT a retrain):
  - Mahalanobis mu/Sigma  : Ledoit-Wolf on clean-TRAIN AE per-channel
                            errors (dataset/fdia_v1/train_z.csv).
  - fusion standardization muf/sdf and 1% fusion threshold thr_f:
                            clean-VAL [bdd_score, maha] (validation/
                            val_scores.csv + dataset/fdia_v1/val_z.csv).

score_vector(window_raw, real_ts): window_raw is (W=10, 168) raw
telemetry in the canonical channel order (current step = last row).
Returns {J, maxLNR, bdd_score, ae_mahalanobis, fusion_score, alarm,
top_channel}.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr
from validation.detect_ae import forward as ae_forward, W as AE_W
from validation.detect_mahalanobis import ae_errors, ledoit_wolf, PARAM_KEYS

BASE = Path(__file__).resolve().parent.parent
FDIA = BASE / "dataset" / "fdia_v1"
AE_WEIGHTS = BASE / "validation" / "detect_ae_weights.npz"
BDD_THRESH = BASE / "config" / "bdd_thresholds_val1pct.json"
FUSION = BASE / "validation" / "detect_final_summary.json"
VAL_SCORES = BASE / "validation" / "val_scores.csv"
CACHE = Path(__file__).resolve().parent / "detector_fit_cache.npz"


class DetectorCore:
    def __init__(self, registry_variant="multiphase", use_cache=True):
        self.model = build_estimator_model(registry_variant)
        self.est = StateEstimator(self.model)

        wz = np.load(AE_WEIGHTS, allow_pickle=True)
        self.P = {k: wz[k] for k in PARAM_KEYS}
        self.mu_ae, self.sd_ae, self.r_c = wz["mu"], wz["sd"], wz["r_c"]

        th = json.load(open(BDD_THRESH))
        self.J99, self.LNR99 = th["J99"], th["LNR99"]
        fw = json.load(open(FUSION))["fusion_weights"]
        self.w_bdd, self.w_maha, self.w_b = fw["bdd"], fw["maha"], fw["b"]

        self.cols = [c for c in pd.read_csv(FDIA / "test_z.csv", nrows=0).columns if c != "timestamp"]

        if use_cache and CACHE.exists():
            c = np.load(CACHE, allow_pickle=True)
            self.mu, self.Sigma, self.muf, self.sdf, self.thr_f, self.maha_thr = (
                c["mu"], c["Sigma"], c["muf"], c["sdf"], float(c["thr_f"]), float(c["maha_thr"]))
        else:
            self._fit_and_cache()
        self.Minv = np.linalg.inv(self.Sigma)

    def _fit_and_cache(self):
        Xtr = pd.read_csv(FDIA / "train_z.csv").drop(columns="timestamp").to_numpy(float)
        Xva = pd.read_csv(FDIA / "val_z.csv").drop(columns="timestamp").to_numpy(float)
        vlab = pd.read_csv(FDIA / "val_labels.csv")["attacked"].to_numpy()
        Etr = ae_errors(self.P, self.mu_ae, self.sd_ae, self.r_c, Xtr)
        tr = Etr[np.isfinite(Etr[:, 0])]
        self.mu = tr.mean(0)
        self.Sigma, _, _ = ledoit_wolf(tr - self.mu)
        Minv = np.linalg.inv(self.Sigma)
        Eva = ae_errors(self.P, self.mu_ae, self.sd_ae, self.r_c, Xva)

        def maha(E):
            s = np.full(len(E), np.nan); fin = np.isfinite(E[:, 0]); d = E[fin] - self.mu
            s[fin] = np.einsum("ij,ij->i", d @ Minv, d); return s
        mv = maha(Eva)
        bdd_v = pd.read_csv(VAL_SCORES)["bdd_score"].to_numpy()
        cvf = (vlab == 0) & np.isfinite(bdd_v) & np.isfinite(mv)
        self.muf = np.array([bdd_v[cvf].mean(), mv[cvf].mean()])
        self.sdf = np.array([bdd_v[cvf].std(), mv[cvf].std()]) + 1e-12
        fin = np.isfinite(bdd_v) & np.isfinite(mv)
        zb = (bdd_v - self.muf[0]) / self.sdf[0]; zm = (mv - self.muf[1]) / self.sdf[1]
        fv = 1.0 / (1.0 + np.exp(-(zb * self.w_bdd + zm * self.w_maha + self.w_b)))
        cvm = (vlab == 0) & fin
        self.thr_f = float(np.quantile(fv[cvm], 0.99))
        self.maha_thr = float(np.quantile(mv[cvm], 0.99))
        np.savez(CACHE, mu=self.mu, Sigma=self.Sigma, muf=self.muf, sdf=self.sdf,
                 thr_f=self.thr_f, maha_thr=self.maha_thr)

    def ae_recon_error(self, window_raw):
        """Per-channel normalized recon error e_c/r_c at the current
        (last) step, from the W=10 window."""
        S = (np.asarray(window_raw, float) - self.mu_ae) / self.sd_ae
        out, _ = ae_forward(self.P, S[None, :, :])
        zhat = out[0] * self.sd_ae + self.mu_ae
        return np.abs(window_raw[-1] - zhat) / self.r_c

    def score_vector(self, window_raw, real_ts):
        window_raw = np.asarray(window_raw, float)
        assert window_raw.shape == (AE_W, len(self.cols)), f"window must be ({AE_W},{len(self.cols)})"
        z = {self.cols[i]: float(window_raw[-1, i]) for i in range(len(self.cols))}
        res = self.est.estimate(z, real_ts)
        st = compute_chi2_and_lnr(self.est, res, z, real_ts)
        J, lnr = float(st["J"]), float(st["lnr"])
        bdd = max(J / self.J99, lnr / self.LNR99)
        e = self.ae_recon_error(window_raw)
        d = e - self.mu
        contrib = d * (self.Minv @ d)
        maha = float(contrib.sum())
        top = self.cols[int(np.argmax(contrib))]
        zb = (bdd - self.muf[0]) / self.sdf[0]; zm = (maha - self.muf[1]) / self.sdf[1]
        fusion = float(1.0 / (1.0 + np.exp(-(zb * self.w_bdd + zm * self.w_maha + self.w_b))))
        return {"J": J, "maxLNR": lnr, "bdd_score": bdd, "ae_mahalanobis": maha,
                "fusion_score": fusion, "alarm": bool(fusion > self.thr_f), "top_channel": top}
