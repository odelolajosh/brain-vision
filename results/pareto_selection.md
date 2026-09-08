### Viability and dominance

Selection rule: viable if measured Raspberry Pi 5 latency < 60 s; Pareto-optimal if not beaten on both accuracy (by more than 1.0 pp) and latency.

| Configuration | F1-noBG (%) | Latency (s) | Tier | Verdict |
|---|---|---|---|---|
| 1D-NN-Fabelo | 70.8 ± 3.2 | 0.103 | Interactive | **Pareto-optimal** |
| 1D-CNN Hu | 73.3 ± 2.0 | 4.346 | Responsive | **Pareto-optimal** |
| 2D-CNN-Simple | 71.4 ± 4.1 | 27.319 | Workflow-compatible | Dominated |
| 1D-NN-Baseline | 68.5 ± 2.5 | 117.233 | Disruptive | Excluded - latency |
| 2D-CNN-Fabelo | 75.9 ± 1.8 | 120.259 | Disruptive | Excluded - latency |
| SF-ViT | not evaluated | 903.443 | Disruptive | Excluded - latency |
| SF-CAF | not evaluated | 1,050.163 | Disruptive | Excluded - latency |
| 2D-CNN-LeeEtAl | not evaluated | 2,688.067 | Disruptive | Excluded - latency |

Recommended configuration: **1D-CNN Hu** at 73.3 ± 2.0% macro F1 excluding background and 4.346 s per whole image.
