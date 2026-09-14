# Per-configuration aggregate

Median ± population std across folds. All metrics in %.

| Configuration (ID) | Model | Tier | Loss | Params | Folds | Val F1-noBG | Test MacroF1 | Test F1-noBG | Test OA | Test TT sens | Test TT F1 | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `1dnn_ce_bal_vpfabelo` | 1D-NN-Baseline | 1 | CE | 17,000,000 | 5 | 67.6 ± 9.1 | 71.3 ± 3.1 | 68.5 ± 2.5 | 79.9 ± 3.8 | 51.4 ± 15.4 | 41.9 ± 5.9 | -- |
| `1dnn_ufl_bal_vpfabelo` | 1D-NN-Baseline | 1 | UFL | 17,000,000 | 5 | 67.6 ± 8.2 | 70.1 ± 1.5 | 69.5 ± 1.6 | 79.6 ± 1.7 | 52.6 ± 10.2 | 42.7 ± 3.9 | -- |
| `1dnnfabelo_ce_bal_vpfabelo` | 1D-NN-Fabelo | 1 | CE | 4,936 | 5 | 70.5 ± 6.3 | 72.6 ± 2.5 | 70.8 ± 3.2 | 79.7 ± 2.0 | 62.4 ± 9.9 | 44.6 ± 7.1 | -- |
| `1dnnfabelo_ce_nobal_vpfabelo` | 1D-NN-Fabelo | 1 | CE | 4,936 | 5 | 73.1 ± 8.6 | 79.9 ± 3.7 | 78.9 ± 4.8 | 84.5 ± 0.9 | 60.7 ± 14.4 | 65.8 ± 14.1 | -- |
| `1dnnfabelo_dl_bal_vpfabelo` | 1D-NN-Fabelo | 1 | DL | 4,936 | 5 | 62.6 ± 7.8 | 62.0 ± 8.0 | 61.8 ± 9.3 | 73.3 ± 4.7 | 27.4 ± 21.0 | 20.5 ± 11.8 | -- |
| `1dnnfabelo_dl_nobal_vpfabelo` | 1D-NN-Fabelo | 1 | DL | 4,936 | 5 | 68.3 ± 5.5 | 74.4 ± 1.3 | 71.6 ± 1.8 | 83.6 ± 0.4 | 32.5 ± 4.3 | 41.9 ± 5.6 | -- |
| `1dnnfabelo_fl_bal_vpfabelo` | 1D-NN-Fabelo | 1 | FL | 4,936 | 5 | 66.6 ± 9.5 | 70.9 ± 3.8 | 69.9 ± 4.2 | 80.0 ± 2.9 | 58.2 ± 12.3 | 44.9 ± 10.7 | -- |
| `1dnnfabelo_fl_nobal_vpfabelo` | 1D-NN-Fabelo | 1 | FL | 4,936 | 5 | 66.3 ± 8.6 | 66.2 ± 1.6 | 66.6 ± 1.4 | 73.7 ± 1.9 | 73.7 ± 3.7 | 33.0 ± 3.5 | -- |
| `1dnnfabelo_ufl_bal_vpfabelo` | 1D-NN-Fabelo | 1 | UFL | 4,936 | 5 | 67.7 ± 9.3 | 68.2 ± 2.6 | 67.4 ± 3.2 | 77.6 ± 1.7 | 59.6 ± 8.7 | 39.7 ± 7.3 | -- |
| `1dnnfabelo_ufl_nobal_vpfabelo` | 1D-NN-Fabelo | 1 | UFL | 4,936 | 4 | 69.7 ± 11.4 | 81.9 ± 1.3 | 81.3 ± 1.4 | 85.6 ± 0.4 | 66.2 ± 6.8 | 69.0 ± 5.1 | -- |
| `1dcnn_ce_bal_vpfabelo` | 1D-CNN-Hu | 2 | CE | 76,824 | 5 | 70.3 ± 5.8 | 74.8 ± 1.9 | 73.3 ± 2.0 | 81.5 ± 1.3 | 65.1 ± 7.7 | 55.6 ± 4.4 | -- |
| `1dcnn_ufl_bal_vpfabelo` | 1D-CNN-Hu | 2 | UFL | 76,824 | 5 | 68.3 ± 5.9 | 73.7 ± 2.6 | 72.8 ± 2.5 | 80.3 ± 1.7 | 70.2 ± 11.3 | 50.4 ± 6.8 | -- |
| `2dcnnfabelo_ce_bal_vpfabelo` | 2D-CNN-Fabelo | 3 | CE | 142,052 | 5 | 70.3 ± 8.9 | 77.6 ± 1.8 | 75.9 ± 1.8 | 83.3 ± 1.1 | 66.5 ± 4.6 | 58.0 ± 6.6 | -- |
| `2dcnnfabelo_ufl_bal_vpfabelo` | 2D-CNN-Fabelo | 3 | UFL | 142,052 | 5 | 70.4 ± 9.4 | 76.1 ± 1.9 | 75.3 ± 2.3 | 81.8 ± 1.4 | 72.4 ± 8.3 | 55.0 ± 5.0 | -- |
| `2dcnn_ce_bal_vpfabelo` | 2D-CNN-LeeEtAl | 3 | CE | 296,580 | 5 | 68.9 ± 10.1 | 73.7 ± 3.2 | 73.1 ± 4.8 | 80.8 ± 1.2 | 58.3 ± 15.6 | 48.3 ± 11.7 | -- |
| `2dcnn_ce_nobal_vpfabelo` | 2D-CNN-LeeEtAl | 3 | CE | 296,580 | 1 | 63.3 ± 0.0 | -- | -- | -- | -- | -- | -- |
| `2dcnn_ufl_bal_vpfabelo` | 2D-CNN-LeeEtAl | 3 | UFL | 296,580 | 5 | 69.8 ± 9.4 | 75.5 ± 2.5 | 75.4 ± 3.5 | 81.7 ± 0.9 | 56.3 ± 10.0 | 51.8 ± 9.9 | -- |
| `2dcnnsimple_ce_bal_vpfabelo` | 2D-CNN-Simple | 3 | CE | 19,644 | 5 | 71.2 ± 10.5 | 72.9 ± 4.0 | 71.4 ± 4.1 | 80.5 ± 2.8 | 68.3 ± 6.2 | 43.5 ± 10.7 | -- |

**Trained but not test-evaluated:** `2dcnn_ce_nobal_vpfabelo`
