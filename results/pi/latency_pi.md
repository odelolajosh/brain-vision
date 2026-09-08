# Inference latency and memory — CPU (Raspberry Pi 5 Model B Rev 1.0)

| field | value |
|---|---|
| Hardware | CPU (Raspberry Pi 5 Model B Rev 1.0) |
| Image | 008-02.npz (480x553, 265,440 px) |
| Patch size | 11 |
| Repetitions | 2 (+1 warm-up) |
| Baseline RSS | 357.4 MB (interpreter + cube loaded) |
| Subsample | 20,000 px, linearly extrapolated |
| Date | 2026-09-07 |

Median of timed repetitions; whole-image inference (every pixel classified).

**Parameter memory** is the float32 weight storage — a property of the architecture. **Peak RSS above baseline** is the additional resident memory used during inference, which for patch models is dominated by the padded cube rather than the model, and so is a deployment-feasibility figure rather than an architectural one.

| Configuration | Params | Param mem (MB) | Type | Per image (s) | px/s | Peak RSS (MB) | RSS over base (MB) | <60 s |
|---|---|---|---|---|---|---|---|---|
| 1D-NN-Fabelo (CE) | 4,936 | 0.02 | pixel | 0.102 | 2,588,968 | 371 | 14 | yes |
| 1D-NN-Baseline (CE) | 17,055,748 | 65.06 | pixel | 117.233 \* | 2,264 | 658 | 300 | **no** |
| 1D-CNN Hu (CE) | 76,824 | 0.29 | pixel | 4.346 | 61,077 | -- | -- | yes |
| 2D-CNN-Simple (CE) | 19,644 | 0.07 | patch | 27.319 | 9,716 | -- | -- | yes |
| 2D-CNN-Fabelo (CE) | 142,052 | 0.54 | patch | 120.259 | 2,207 | -- | -- | **no** |
| 2D-CNN-LeeEtAl (CE) | 296,580 | 1.13 | patch | 2688.067 \* | 99 | -- | -- | **no** |
| SpectralFormer-ViT (CE) | 198,197 | 0.76 | pixel | 903.443 \* | 294 | -- | -- | **no** |
| SpectralFormer-CAF (CE) | 198,197 | 0.76 | pixel | 1050.163 \* | 253 | -- | -- | **no** |

\* extrapolated from a random subsample — estimate.

Peak RSS is attributable only to the first configuration timed in a process (`ru_maxrss` is a non-resettable high-water mark); re-run one configuration per process for per-config memory.
