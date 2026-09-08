# Inference latency and memory — MPS (Apple Silicon, arm64)

| field | value |
|---|---|
| Hardware | MPS (Apple Silicon, arm64) |
| Image | 008-02.npz (480x553, 265,440 px) |
| Patch size | 11 |
| Repetitions | 5 (+1 warm-up) |
| Baseline RSS | 496.9 MB (interpreter + cube loaded) |
| Date | 2026-09-07 |

Median of timed repetitions; whole-image inference (every pixel classified).

**Parameter memory** is the float32 weight storage — a property of the architecture. **Peak RSS above baseline** is the additional resident memory used during inference, which for patch models is dominated by the padded cube rather than the model, and so is a deployment-feasibility figure rather than an architectural one.

| Configuration | Params | Param mem (MB) | Type | Per image (s) | px/s | Peak RSS (MB) | RSS over base (MB) | <60 s |
|---|---|---|---|---|---|---|---|---|
| 1D-NN-Fabelo (CE) | 4,936 | 0.02 | pixel | 0.018 | 14,317,473 | 592 | 95 | yes |
| 1D-NN-Baseline (CE) | 17,055,748 | 65.06 | pixel | 2.658 | 99,856 | -- | -- | yes |
| 1D-CNN Hu (CE) | 76,824 | 0.29 | pixel | 0.114 | 2,328,048 | -- | -- | yes |
| 2D-CNN-Simple (CE) | 19,644 | 0.07 | patch | 3.704 | 71,664 | -- | -- | yes |
| 2D-CNN-Fabelo (CE) | 142,052 | 0.54 | patch | 5.243 | 50,623 | -- | -- | yes |
| 2D-CNN-LeeEtAl (CE) | 296,580 | 1.13 | patch | 161.408 | 1,644 | -- | -- | **no** |
| SpectralFormer-ViT (CE) | 198,197 | 0.76 | pixel | 38.082 | 6,970 | -- | -- | yes |
| SpectralFormer-CAF (CE) | 198,197 | 0.76 | pixel | 43.418 | 6,114 | -- | -- | yes |

Peak RSS is attributable only to the first configuration timed in a process (`ru_maxrss` is a non-resettable high-water mark); re-run one configuration per process for per-config memory.
