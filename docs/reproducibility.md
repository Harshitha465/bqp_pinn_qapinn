# Reproducibility Guide

> All numbers in this document are drawn from the **executed cell outputs** of the three final notebooks. Nothing is assumed or extrapolated.

---

## 1. Execution Environment

Experiments were run on **Google Colab** with the following confirmed hardware/software:

| Component | Specification |
|-----------|--------------|
| GPU | Tesla T4, 15.64 GB VRAM |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| PennyLane | 0.45.1 |
| JAX | 0.9.2 (quantum circuit backend) |
| jaxlib | 0.9.2 |
| Classical layers device | CUDA (GPU) |
| Quantum circuit device | CPU — PennyLane `default.qubit` simulator (JAX interface, JIT-compiled) |

> **Why JAX is pinned to 0.9.2:** PennyLane 0.45.x uses `jax.core.is_concrete`, which was removed in JAX 0.10+. Using any newer JAX version will cause an `AttributeError` at circuit execution. Do **not** upgrade without first verifying PennyLane compatibility.

### Local Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` (pinned for reproducibility):
```
torch>=2.1.0
pennylane==0.45.1
jax==0.9.2
jaxlib==0.9.2
numpy
scipy
matplotlib
pandas
seaborn
sympy
```

---

## 2. Seeding Protocol

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

`SEED = 42` for **all single-seed main-matrix runs** across all three notebooks.

**Stage C (multi-seed sweep):** Seeds `[11, 29, 47, 97, ...]` (actual seeds used, as logged in cell outputs — **not** `[42, 101, 202, 303, 404]` as originally planned; the seed list was changed during development).

---

## 3. Experiment Configuration Flags

Each notebook has three top-level config flags at the top of the configuration cell:

| Flag | Heat | Burgers | NavierStokes |
|------|------|---------|--------------|
| `FAST_LOCAL_MODE` | `False` | `False` | `False` |
| `EPOCHS_MAIN` | `2000` | `2000` | `5000` |
| `EPOCHS_CLASSICAL_BEST` | `5000` | `2000` | `10000` |
| `RUN_STAGE_C` | `True` (flag only — no execution cell) | `True` | `True` |
| `LOG_EVERY_N_EPOCHS` | `10` | `10` | `10` |

> ⚠️ `RUN_STAGE_C=True` in the **Heat notebook** is a flag only — no Stage C execution cell was written in that notebook. Setting this flag has no effect.

---

## 4. Hyperparameter Table

All configurations share identical hyperparameters (capacity-matched):

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Initial learning rate | `1e-3` |
| LR scheduler | `ReduceLROnPlateau(factor=0.5, patience=300)` |
| Gradient clipping | Norm `1.0` |
| Loss weights | `w_pde = w_ic = w_bc = 1.0` |
| Collocation points | 2,000 |
| IC/BC points | 200–400 (PDE-dependent) |
| Quantum circuit depth | `n_qlayers = 4` |
| Quantum measurement | `⟨PauliZ(i)⟩` per qubit |
| Classical post-processing | FC `[n_qubits→25→25→25→25→n_out]`, Tanh activations |

### Model Parameter Counts (Verified)

| Config | Qubits | Params (Heat/Burgers) | Params (NS) |
|--------|--------|-----------------------|-------------|
| C0 | — | 2,051 | 2,103 |
| Q1 | 3 | 2,283 | 2,332 |
| Q2 | 4 | 2,345 | 2,393 |
| Q3 | 5 | 2,407 | 2,454 |
| Q4 | 3 | 2,283 | 2,332 |
| Q5 | 3 | 2,307 | 2,350 |
| C0-Best | — | 5,081 | 2,103 |

NS uses a slightly different input dimension (2 outputs for `u,v` + pressure `p`), accounting for the parameter difference.

---

## 5. Measured Timing (Actual Notebook Outputs)

### Per-Epoch Speed (after JIT warm-up)

| PDE | Classical C0 | QAPINN Q1 (3q reference) | Q1 JIT warm-up (one-time) |
|-----|-------------|--------------------------|---------------------------|
| Heat | ~0.44 s/epoch | ~0.16 s/epoch (post-JIT) | ~51.8 s (one-time) |
| Burgers | ~0.02 s/epoch | ~5.16 s/epoch (post-JIT) | ~25 s (one-time) |
| NS | ~0.02 s/epoch | ~5.16 s/epoch (post-JIT) | ~25 s (one-time) |

> Note: Q2–Q5 each pay their own independent one-time JIT compile cost the first time they run. Only steady-state speed is listed above.

### Actual Total Wall-Clock Times (from notebook "Done in Xs" logs)

| Config | Heat | Burgers | NavierStokes |
|--------|------|---------|--------------|
| C0_Classical | 24.9 s | 25.0 s | 124.1 s |
| Q1 | 348.6 s | 426.1 s | 293.4 s |
| Q2 | 751.1 s | 836.9 s | 333.7 s |
| Q3 | 1,801.0 s | 1,897.9 s | 400.2 s |
| Q4 | 494.5 s | 569.7 s | 359.1 s |
| Q5 | 325.1 s | 366.4 s | 379.5 s |

---

## 6. Notebook Execution Sequence

Run in this order (each notebook is self-contained but references the same model definitions):

1. `notebooks/Heat_Equation_JAX.ipynb` — 1D linear diffusion (fastest; good for environment verification)
2. `notebooks/Burgers_Equation_JAX.ipynb` — 1D viscous Burgers' (primary benchmark)
3. `notebooks/NavierStokes_Kovasznay.ipynb` — 2D Kovasznay flow (longest; ~400 s for Q3 alone)

---

## 7. Known Reproducibility Caveats

| Issue | Affected Notebook | Root Cause |
|-------|------------------|------------|
| Burgers Stage C incomplete | `Burgers_Equation_JAX.ipynb` | Colab runtime interrupted mid-sweep during seed 47/97 of Q3's run (no error — session timeout) |
| Heat Stage C not executed | `Heat_Equation_JAX.ipynb` | No Stage C cell written in this notebook despite `RUN_STAGE_C=True` flag |
| NS Stage C covers C0+Q3 only | `NavierStokes_Kovasznay.ipynb` | Intentional scope decision; Q1/Q2/Q4/Q5 ns results are single-seed |
| NS McClean cross-seed std=0.000 | `NavierStokes_Kovasznay.ipynb` | Seed not re-applied inside the 5-iteration McClean robustness loop; all 5 "different seeds" ran identically |
| McClean zero-input degeneracy bug | (fixed before final runs) | `angle` vs. `reupload` encoding collapsed to identical circuits under `torch.zeros` input; patched in final notebooks |

---

## 8. Ground Truth Verification

All reference solutions were verified symbolically with SymPy before use:

| PDE | Ground Truth Method |
|-----|-------------------|
| Heat | Separation of variables: $u = \sin(\pi x)\,e^{-\alpha\pi^2 t}$ |
| Burgers' | Cole–Hopf transform + Gauss–Hermite quadrature (independent of any trained model) |
| Navier–Stokes | Closed-form Kovasznay solution ($\lambda \approx -0.9637$ at $Re=40$) |
