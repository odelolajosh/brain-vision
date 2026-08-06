# brainvision — Streamlit Demo

Intraoperative HSI brain tumour classification demonstrator.
Imports directly from the `brainvision` package — no duplicated code.

## Run locally (development)

```bash
# From the repo root
streamlit run demo/app.py
```

## Deploy to Raspberry Pi 5

Copy only what the Pi needs — no training code, no notebooks:

```
brainvision/          ← package (models, preprocessing, constants)
demo/app.py
demo/requirements_demo.txt
checkpoints/
  1dnnfabelo_ce_bal_fold2_vpfabelo.pt   ← best checkpoint (~20 KB)
```

Install on Pi:

```bash
# CPU-only PyTorch for Pi (much smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r demo/requirements_demo.txt

# Run — accessible from any device on the same network
streamlit run demo/app.py --server.address 0.0.0.0
# → open http://<pi-ip>:8501 in browser
```

## Import map

| app.py uses | comes from |
|---|---|
| `CLASS_COLORS`, `CLASS_NAMES`, `N_BANDS`, `N_CLASSES` | `brainvision/constants.py` |
| `preprocess()` | `brainvision/preprocessing.py` |
| `FabeloDNN` | `brainvision/models/fabelo_dnn.py` |
| `BaselineDNN` | `brainvision/models/baseline_dnn.py` |
| `HuCNN1D` | `brainvision/models/hu_1dcnn.py` |
| `Fabelo2DCNN` | `brainvision/models/fabelo_2dcnn.py` |
| `Lee2DCNN` | `brainvision/models/lee_2dcnn.py` |
| `Simple2DCNN` | `brainvision/models/simple_2dcnn.py` |

## Fixing import errors

If a class name doesn't match, open the relevant model file and check
the class name at the top, then update the import in `demo/app.py`.
Example: if `hu_1dcnn.py` defines `class CNN1D` not `class HuCNN1D`:

```python
# demo/app.py line ~20
from brainvision.models.hu_1dcnn import CNN1D as HuCNN1D
```

## Notes

- **1D-NN-Fabelo recommended for Pi 5** — 4,936 params, near-instant inference
- 2D patch models work but are slow on Pi (~minutes per image)
- Raw 826-band cubes: enable "Raw cube" toggle — runs full preprocessing pipeline
- Processed 128-band cubes: toggle off — only normalisation applied
