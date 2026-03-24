# ASTC weight probe
Base test accuracy: 0.9722
## Whole-model schemes
- fp16: acc=0.9722, drop_pp=0.00, bpw=16.000, logit_rmse=0.001
- int4_6x6: acc=0.9694, drop_pp=0.28, bpw=5.241, logit_rmse=0.163
- astc_4x4_minshift: acc=0.9806, drop_pp=-0.83, bpw=9.089, logit_rmse=0.175
- astc_6x6_minshift: acc=0.9444, drop_pp=2.78, bpw=4.288, logit_rmse=1.453
- astc_8x8_minshift: acc=0.7306, drop_pp=24.17, bpw=2.317, logit_rmse=3.027
- astc_6x6_raw: acc=0.4000, drop_pp=57.22, bpw=3.812, logit_rmse=20.169

## 6x6 per-layer sensitivity
- layer 0 shape=(64, 128): acc_if_only_layer_replaced=0.9611, drop_pp=1.11, bpw=4.254, rmse=0.0445
- layer 1 shape=(128, 128): acc_if_only_layer_replaced=0.9500, drop_pp=2.22, bpw=4.254, rmse=0.0375
- layer 2 shape=(128, 10): acc_if_only_layer_replaced=0.9611, drop_pp=1.11, bpw=4.950, rmse=0.0570

## SVD side-path + 6x6 ASTC
- rank 1: acc=0.9639, drop_pp=0.83, bpw=4.651, logit_rmse=1.306
- rank 2: acc=0.9500, drop_pp=2.22, bpw=5.014, logit_rmse=1.125
- rank 4: acc=0.9361, drop_pp=3.61, bpw=5.739, logit_rmse=1.049
- rank 8: acc=0.9639, drop_pp=0.83, bpw=7.189, logit_rmse=0.722
- rank 16: acc=0.9722, drop_pp=0.00, bpw=9.578, logit_rmse=0.542
