# Results — Interpretation & Caveats

> All metrics in this directory are drawn from executed notebook cell outputs (SEED=42 single-seed main matrix). Multi-seed Stage C results are partial.

---

## Directory Structure

Each PDE subdirectory contains:

| File | Description |
|------|-------------|
| `metrics.csv` | Per-configuration performance metrics (Rel L2, PDE loss, params, time) |
| `training_loss.png` | Training loss curves for C0 and Q1–Q5 |
| `comparison_bars.png` | Relative L2 error bar chart across configurations |
| `fourier_spectrum.png` | Fourier spectrum analysis of quantum circuit outputs |
| `activation_maps.png` | Layer-1 pre-activation heatmaps |
| `master_performance_table.png` | Summary performance table visualisation |

---

## Statistical Significance

Only the following results are backed by multi-seed Welch's t-test (Bonferroni-corrected α = 0.01):

| PDE | Config | p-value | Status |
|-----|--------|---------|--------|
| Burgers' | Q1 (3q, angle, basic) | 0.0092 | ✅ Significant |
| Navier–Stokes | Q3 (5q, angle, basic) | 0.0037 | ✅ Significant |

All other results are **single-seed (SEED=42)** and should be treated as suggestive, not confirmed.

---

## Known Caveats

- **Burgers' Stage C incomplete:** Q2, Q3, Q5 multi-seed sweeps were runtime-interrupted (Colab session timeout). Only Q1 and Q4 have valid 5-seed statistics.
- **Heat equation — no multi-seed test:** All Heat metrics are single-seed. The `RUN_STAGE_C=True` flag exists in the notebook but no Stage C execution cell was written.
- **NS Stage C covers C0 and Q3 only:** Other NS configs (Q1, Q2, Q4, Q5) are single-seed.
- **NS McClean cross-seed std=0.000:** Seed was not re-applied per run in the robustness loop; treat NS McClean gradient-variance numbers as single-seed only.
- **Measurement basis:** Only expectation values ⟨Z⟩ per qubit were tested. Probability-vector measurements were not evaluated.

---

## Reproducing Results

See [`docs/reproducibility.md`](../docs/reproducibility.md) for the complete environment specification, seeding protocol, and hyperparameter table. Run notebooks in order:

1. `notebooks/Heat_Equation.ipynb`
2. `notebooks/Burgers_Equation.ipynb`
3. `notebooks/NavierStokes_Kovasznay.ipynb`
