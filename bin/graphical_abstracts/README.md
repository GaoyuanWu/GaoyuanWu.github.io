# Graphical abstract generators

Scripts that produce `assets/img/graphical_abstracts/<paper-id>.webp` (animated) and `.png` (still,
served to visitors who prefer reduced motion). All figures are 1200×620 px.

| Paper id                  | Data / solver script | Renderer         | Notes                                                                       |
| ------------------------- | -------------------- | ---------------- | --------------------------------------------------------------------------- |
| `wu2023framework`         | `mm_run.py`          | `r_framework.py` | Real JaxSSO run of the Mannheim Multihalle example (200 GD iterations)      |
| `wu2024jaxsso`            | `nn_run.py`          | `r_jaxsso.py`    | MLP → shell heights, trained with JaxSSO strain energy as the loss          |
| `wu2023gridshells`        | `grid_run.py`        | `r_grid.py`      | Illustrative 20 m gridshell: constrained FDM vs FEA-based shape opt.        |
| `wu2026bayes`             | —                    | `r_bayes.py`     | Paper's SPH snapshots + schematic Pareto panel; stats from the abstract     |
| `wu2025sph`               | —                    | `r_sph.py`       | Cross-sections redrawn from the paper figure; schematic regular waves       |
| `smith2025neural`         | —                    | `r_nn_hypar.py`  | Frames of `Side_Hypar.gif` (SPH) + schematic ANN pipeline                   |
| `pawitan2026experimental` | —                    | `r_pawitan.py`   | Uplift-force traces + flume photos digitized from the paper's Fig. 13       |
| `jmse13020245`            | —                    | `r_jmse.py`      | Hypar faces built from the paper's Rn definition; coefficients from Table 4 |

Setup (run from a scratch directory, not inside the site):

```bash
python -m venv venv && . venv/bin/activate
git clone https://github.com/GaoyuanWu/JaxSSO ref/JaxSSO
pip install -e ref/JaxSSO "jax==0.4.35" "jaxlib==0.4.35" "numpy==2.1.3" "scipy==1.14.1" matplotlib pandas pillow
mkdir -p ga look
```

`mm_run.py` is generated from `ref/JaxSSO/Examples/Shells_Mannheim_Multihalle_Shape.ipynb` and expects the
example data at `ref/JaxSSO/Examples/Data/Mannheim_Quad/`. `r_bayes.py` expects `ga/sph_rect.png` and
`ga/sph_opt.png`, cropped from `assets/img/publication_preview/Multi_press_visualization.png`. `r_pawitan.py` expects `ga/pawitan_series.npy` (digitized curves) and `ga/pawitan/ins_*.png` (photo crops). The renderers
write to `ga/out/`; copy the results into `assets/img/graphical_abstracts/`. Newer JAX/SciPy releases make
JaxSSO's sparse solve return NaN, hence the pins above. The renderers use the Inter font when it is installed.
