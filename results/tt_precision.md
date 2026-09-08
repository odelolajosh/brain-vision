### Tumour tissue — precision, recall and F1

Median ± population standard deviation across folds, computed from the saved test confusion matrices. Each fold is a distinct model evaluated on the same fixed test partition, so per-fold metrics are aggregated rather than the matrices being pooled.

| Configuration | Precision (%) | Recall (%) | Specificity (%) | F1 (%) |
|---|---|---|---|---|
| `1dcnn_ce_bal_vpfabelo` | 43.3 ± 6.6 | 65.1 ± 7.7 | 95.2 ± 1.3 | 55.6 ± 4.4 |
| `1dcnn_ce_nobal_vpfabelo` | 75.0 ± 5.6 | 69.6 ± 6.2 | 98.7 ± 0.5 | 71.6 ± 3.9 |
| `1dcnn_ufl_bal_vpfabelo` | 39.3 ± 9.9 | 70.2 ± 11.3 | 93.7 ± 2.4 | 50.4 ± 6.8 |
| `1dcnn_ufl_nobal_vpfabelo` | 80.8 ± 6.9 | 56.4 ± 6.2 | 99.1 ± 0.3 | 70.8 ± 3.8 |
| `1dnn_ce_bal_vpfabelo` | 39.0 ± 10.4 | 51.4 ± 15.4 | 95.3 ± 5.7 | 41.9 ± 5.9 |
| `1dnn_ufl_bal_vpfabelo` | 35.9 ± 4.9 | 52.6 ± 10.2 | 94.6 ± 2.3 | 42.7 ± 3.9 |
| `1dnnfabelo_ce_bal_vpfabelo` | 36.4 ± 12.4 | 62.4 ± 9.9 | 94.2 ± 3.8 | 44.6 ± 7.1 |
| `1dnnfabelo_ce_nobal_vpfabelo` | 71.9 ± 17.7 | 60.7 ± 14.4 | 98.6 ± 1.3 | 65.8 ± 14.1 |
| `1dnnfabelo_dl_bal_vpfabelo` | 16.4 ± 6.5 | 27.4 ± 21.0 | 89.7 ± 4.2 | 20.5 ± 11.8 |
| `1dnnfabelo_dl_nobal_vpfabelo` | 41.5 ± 20.0 | 32.5 ± 4.3 | 96.5 ± 1.5 | 41.9 ± 5.6 |
| `1dnnfabelo_fl_bal_vpfabelo` | 36.2 ± 16.2 | 58.2 ± 12.3 | 93.9 ± 4.0 | 44.9 ± 10.7 |
| `1dnnfabelo_fl_nobal_vpfabelo` | 21.2 ± 3.0 | 73.7 ± 3.7 | 84.1 ± 2.9 | 33.0 ± 3.5 |
| `1dnnfabelo_ufl_bal_vpfabelo` | 27.0 ± 8.4 | 59.6 ± 8.7 | 90.6 ± 2.6 | 39.7 ± 7.3 |
| `1dnnfabelo_ufl_nobal_vpfabelo` | 72.0 ± 4.0 | 61.8 ± 6.5 | 98.6 ± 0.2 | 66.8 ± 5.1 |
| `2dcnn_ce_bal_vpfabelo` | 39.4 ± 10.3 | 58.3 ± 15.6 | 94.0 ± 1.6 | 48.3 ± 11.7 |
| `2dcnn_ufl_bal_vpfabelo` | 43.5 ± 10.9 | 56.3 ± 10.0 | 95.2 ± 2.2 | 51.8 ± 9.9 |
| `2dcnnfabelo_ce_bal_vpfabelo` | 50.9 ± 11.5 | 66.5 ± 4.6 | 96.2 ± 2.1 | 58.0 ± 6.6 |
| `2dcnnfabelo_ufl_bal_vpfabelo` | 46.8 ± 7.6 | 72.4 ± 8.3 | 96.0 ± 2.0 | 55.0 ± 5.0 |
| `2dcnnsimple_ce_bal_vpfabelo` | 31.9 ± 16.7 | 68.3 ± 6.2 | 91.5 ± 3.3 | 43.5 ± 10.7 |
| `spectralformer_caf_ce_bal_vpfabelo` | 32.9 ± 5.5 | 64.1 ± 1.7 | 92.3 ± 2.0 | 43.5 ± 5.1 |
| `spectralformer_vit_ce_bal_vpfabelo` | 31.1 ± 7.6 | 55.7 ± 7.5 | 91.8 ± 2.0 | 41.9 ± 8.2 |
| `spectralformer_vit_ufl_bal_vpfabelo` | 27.6 ± 6.2 | 59.5 ± 7.6 | 91.0 ± 1.9 | 37.7 ± 6.9 |
