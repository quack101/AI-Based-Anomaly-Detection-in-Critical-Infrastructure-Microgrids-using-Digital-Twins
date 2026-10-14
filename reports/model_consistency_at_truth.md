# Model consistency at the TRUE state

Script: `validation/model_consistency_at_truth.py`. Every number below comes from this one run unless noted.

Points: 120 (same union as validation/exp_all91_physics.py), P0 physics.

## 3a -- J at truth vs J at the estimate

- J_truth[real]: mean=1.1493e+12  median=1.6126e+12
- J_truth[pseudo]: mean=8.8156e+02  median=7.0707e+02
- J_truth[regzi]: mean=3.4657e+01  median=2.4569e+01
- J_truth[total] mean=1.1493e+12 median=1.6126e+12
- J_at_estimate (from exp_all91_physics_results.csv, P0) mean=3.0103e+04 median=2.9623e+04
- J_truth / J_at_estimate ratio, mean of per-point ratios: 51472446.6511

## 3b -- share of J_truth by channel group

- meter_Q: 1.3791e+14  (100.00% of total J_truth)
- meter_P: 3.1805e+07  (0.00% of total J_truth)
- HEAD_P: 3.4370e+06  (0.00% of total J_truth)
- pseudo: 1.0579e+05  (0.00% of total J_truth)
- regzi: 4.1589e+03  (0.00% of total J_truth)
- HEAD_Q: 3.2384e-03  (0.00% of total J_truth)
- V_magnitude: 1.3854e-24  (0.00% of total J_truth)
- zero-injection residual at truth: max|r0|=0.000000  mean|r0|=0.000000

## 3c -- top 15 REAL channels by |mean(r0)/sigma|

- M004_Q: mean(r0/sigma)=-953589.7559  std=491872.9566  sign=-  mean(r0)=-190717.9050
- M004_P: mean(r0/sigma)=-392.3588  std=301.5006  sign=-  mean(r0)=-156.9445
- M040_P: mean(r0/sigma)=103.7647  std=101.3185  sign=+  mean(r0)=20.7531
- HEAD_P_B: mean(r0/sigma)=96.8274  std=19.3494  sign=+  mean(r0)=931.5865
- HEAD_P_A: mean(r0/sigma)=95.3103  std=20.5994  sign=+  mean(r0)=1394.6506
- HEAD_P_C: mean(r0/sigma)=95.0229  std=19.0773  sign=+  mean(r0)=1130.6369
- M008_P: mean(r0/sigma)=0.0374  std=0.0293  sign=+  mean(r0)=40.5923
- M015_P: mean(r0/sigma)=0.0352  std=0.0339  sign=+  mean(r0)=40.0380
- M011_P: mean(r0/sigma)=0.0317  std=0.0318  sign=+  mean(r0)=36.1376
- M009_P: mean(r0/sigma)=0.0304  std=0.0340  sign=+  mean(r0)=31.4246
- M048_Q: mean(r0/sigma)=0.0284  std=0.0284  sign=+  mean(r0)=33.3334
- M025_P: mean(r0/sigma)=0.0283  std=0.0306  sign=+  mean(r0)=33.8583
- M048_P: mean(r0/sigma)=0.0282  std=0.0283  sign=+  mean(r0)=71.4285
- M020_P: mean(r0/sigma)=0.0234  std=0.0276  sign=+  mean(r0)=33.3177
- M008_Q: mean(r0/sigma)=0.0232  std=0.0182  sign=+  mean(r0)=20.2962

## 3d -- HEAD_P_A/B/C: r0, percent of head P, linear fit vs P_head, hour dependence

### HEAD_P_A
- h_true (kW): mean=0.00e+00  std=0.00e+00
- r0 (kW): mean=1394.651  std=301.426  mean %head_P=undefined (h_true~0)
- fit r0 = a + b*P_head: SKIPPED -- h_true has ~zero variance (std=0.00e+00, mean=0.00e+00); r0 is essentially -z0 itself, not a function of h_true
- by hour-of-day, mean r0 (kW): {0: 1130.762, 1: 1039.602, 2: 968.058, 3: 984.655, 4: 1042.324, 5: 1317.183, 6: 1458.262, 7: 1627.704, 8: 1629.876, 9: 1544.336, 10: 1513.835, 11: 1631.033, 12: 1390.939, 13: 1312.784, 14: 1366.409, 15: 1419.335, 16: 1599.833, 17: 1806.403, 18: 1977.66, 20: 1676.754, 21: 1691.709, 22: 1465.514, 23: 1258.959}

### HEAD_P_B
- h_true (kW): mean=0.00e+00  std=0.00e+00
- r0 (kW): mean=931.587  std=186.163  mean %head_P=undefined (h_true~0)
- fit r0 = a + b*P_head: SKIPPED -- h_true has ~zero variance (std=0.00e+00, mean=0.00e+00); r0 is essentially -z0 itself, not a function of h_true
- by hour-of-day, mean r0 (kW): {0: 750.239, 1: 693.401, 2: 669.776, 3: 684.347, 4: 710.311, 5: 917.721, 6: 978.185, 7: 1111.232, 8: 1089.383, 9: 958.453, 10: 1024.705, 11: 1020.301, 12: 923.213, 13: 907.373, 14: 958.24, 15: 981.675, 16: 1012.385, 17: 1131.366, 18: 1283.294, 20: 1128.684, 21: 1188.389, 22: 961.412, 23: 813.571}

### HEAD_P_C
- h_true (kW): mean=0.00e+00  std=0.00e+00
- r0 (kW): mean=1130.637  std=226.993  mean %head_P=undefined (h_true~0)
- fit r0 = a + b*P_head: SKIPPED -- h_true has ~zero variance (std=0.00e+00, mean=0.00e+00); r0 is essentially -z0 itself, not a function of h_true
- by hour-of-day, mean r0 (kW): {0: 919.943, 1: 851.09, 2: 805.249, 3: 812.766, 4: 864.426, 5: 1109.962, 6: 1167.847, 7: 1324.225, 8: 1284.143, 9: 1253.486, 10: 1278.907, 11: 1271.102, 12: 1127.921, 13: 1062.898, 14: 1181.854, 15: 1174.294, 16: 1222.716, 17: 1405.098, 18: 1487.938, 20: 1339.896, 21: 1498.01, 22: 1215.281, 23: 1034.933}

## 3e -- top 15 (same metric) at the ESTIMATE x_hat

- HEAD_P_B: mean(nr)=96.9934  std=19.3359  sign=+
- HEAD_P_A: mean(nr)=95.2797  std=20.7867  sign=+
- HEAD_P_C: mean(nr)=95.1528  std=19.1603  sign=+
- M024_V: mean(nr)=8.0205  std=1.4789  sign=+
- M023_V: mean(nr)=7.6642  std=1.5582  sign=+
- M021_V: mean(nr)=7.4157  std=1.4643  sign=+
- M022_V: mean(nr)=7.1809  std=1.1744  sign=+
- M042_V: mean(nr)=7.1699  std=1.1850  sign=+
- M047_V: mean(nr)=6.8434  std=1.3855  sign=+
- M026_V: mean(nr)=6.6545  std=1.5048  sign=+
- M027_V: mean(nr)=6.5372  std=1.4055  sign=+
- REGZI_160_A_Q: mean(nr)=-6.1858  std=0.1766  sign=-
- M025_V: mean(nr)=6.1064  std=1.1871  sign=+
- REGZI_160r_A_Q: mean(nr)=5.8607  std=0.1577  sign=+
- M041_V: mean(nr)=5.8123  std=1.3567  sign=+

## Step 5 -- diagnostic what-if: HEAD_P sigma x100, in-process only, 10 stratified points

(mutates a fresh model's meas_entries dict in memory; nothing written to the registry file)

  i     J_normal  J_sigma100  converged_normal  converged_sigma100  err_median_normal  err_p95_normal  err_max_normal  err_median_sigma100  err_p95_sigma100  err_max_sigma100
 46 18100.336028 1485.978689              True                True           0.024117        0.042427        0.050754             0.024117          0.042427          0.050754
 94 16188.519455 1095.954527              True                True           0.019639        0.035818        0.043003             0.019639          0.035818          0.043003
154 14297.401062  865.036765              True                True           0.014917        0.031048        0.036442             0.014917          0.031048          0.036442
166 14182.146713 1063.162838              True                True           0.014296        0.031641        0.037279             0.014296          0.031641          0.037279
181 13734.609029  926.023225              True                True           0.013685        0.031004        0.038330             0.013685          0.031004          0.038330
216 12380.590195  889.883969              True                True           0.014330        0.029713        0.036805             0.014330          0.029713          0.036805
285 16009.447472 1068.288258              True                True           0.015509        0.035056        0.041864             0.015509          0.035056          0.041864
288 17851.566779 1014.286944              True                True           0.015730        0.029942        0.036468             0.015730          0.029942          0.036468
359 31963.163657 1507.907631              True                True           0.022470        0.044353        0.048137             0.022470          0.044353          0.048137
368 40211.983393 1574.087890              True                True           0.027231        0.044483        0.050653             0.027231          0.044483          0.050653

- mean J: normal=1.9492e+04  sigma100=1.1491e+03
- median v-err: normal=0.0156  sigma100=0.0156
- p95 v-err: normal=0.0333  sigma100=0.0333
- max v-err: normal=0.0401  sigma100=0.0401

## Runtime

- truth+residual build (single process, 120 points): 19.5s
- estimator solves (120 calls, <= 4 workers): 578.0s
- what-if (10 points x 2 solves, sequential): 227.1s
- total script runtime: 825.1s

