# Quantum-Assisted Physics-Informed Neural Networks (QAPINN) for CFD

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Report](https://img.shields.io/badge/Report-PDF-blue)](REPORT.pdf)
[![Presentation](https://img.shields.io/badge/Presentation-PPTX-orange)](PRESENTATION.pptx)

> **BQP WISER Global Quantum+AI 2026** — Research repository investigating *when, why, and how* substituting a variational quantum circuit into a classical PINN changes its learning dynamics for PDE-governed fluid systems.

**Team Members**
- Lingerker Harshita Bai
- Amavasya Manoj Kumar Reddy
- Ansh Goel

---

## Overview

Physics-Informed Neural Networks (PINNs) solve partial differential equations by embedding governing equations directly into a loss function, using automatic differentiation to penalise residuals [1]. This project evaluates a hybrid architecture — the **Quantum-Assisted PINN (QAPINN)** — where the first hidden layer of a classical PINN is replaced by a Variational Quantum Circuit (VQC), following Shah, Lineswala, and Chopra's QA-PINN benchmarking approach for CFD [2].

**The goal is not to prove QAPINN beats classical PINN.** It is to characterise **when, why, and how** the quantum substitution changes learning dynamics — using Fourier-spectrum expressivity analysis [3], McClean-style barren-plateau gradient-variance scans, weight-movement diagnostics, and activation maps — across three PDE benchmarks of increasing complexity.

In addition to evaluating predictive performance, the project investigates the impact of different quantum circuit designs—including qubit count, data encoding strategies, entanglement structures, and circuit architectures. The repository includes the complete implementation, experimental results, technical report, presentation slides, and documentation required to reproduce the study.

---

## Objectives

The objectives of this project are to:

- Investigate how the introduction of a quantum layer influences the learning process of Physics-Informed Neural Networks (PINNs).
- Compare the learning behavior and predictive performance of Classical PINNs and Quantum-Assisted PINNs (QAPINNs) across benchmark partial differential equations.
- Evaluate the influence of quantum circuit design choices, including qubit count, data encoding strategies, and variational ansatz architectures.
- Analyze learning dynamics using Explainable AI (XAI) techniques, including Fourier spectrum analysis, gradient variance analysis, and activation map visualization.
- Assess the effectiveness of QAPINNs using quantitative performance metrics such as Relative L2 Error, PDE residual error, training time, and model complexity.
- Develop insights and design guidelines for constructing problem-specific quantum-enhanced architectures for solving partial differential equations.

---

## PDE Benchmarks

| # | PDE | Type | Domain |
|---|-----|------|--------|
| 1 | **1D Viscous Burgers' Equation** | Nonlinear convective-diffusive | $x \in [-1,1],\ t \in [0,1]$, $\nu = 0.01/\pi$ |
| 2 | **1D Heat Equation** | Linear diffusion | $x \in [-1,1],\ t \in [0,1]$, $\alpha = 0.1$ |
| 3 | **2D Kovasznay Flow (Navier–Stokes)** | Steady, coupled, nonlinear vector-valued | $[-0.5,1]\times[-0.5,1.5]$, $Re = 40$ |

All ground-truth solutions are verified analytically (Cole–Hopf for Burgers', separation-of-variables for heat, exact Kovasznay formula for NS — all cross-checked with SymPy).

---

## Experiment Matrix

Six configurations, architecturally identical across all three PDE notebooks (~2,100–2,450 trainable parameters each):

| ID | Kind | Qubits | Encoding | Ansatz |
|----|------|--------|----------|--------|
| **C0** | Classical PINN | — | — | FC [25-25-25-25] |
| **Q1** | QAPINN | 3 | angle | BasicEntanglerLayers |
| **Q2** | QAPINN | 4 | angle | BasicEntanglerLayers |
| **Q3** | QAPINN | 5 | angle | BasicEntanglerLayers |
| **Q4** | QAPINN | 3 | reupload | BasicEntanglerLayers |
| **Q5** | QAPINN | 3 | angle | StronglyEntanglingLayers |

Each PDE also trains a `C0_Best_Classical` variant (wider network / longer budget) as a practical classical ceiling.

**QAPINN architecture:** `input_proj (2 → n_qubits) [GPU] → VQC [CPU/GPU via JAX] → classical post-processing [25-25-25-25] → output`

---

## Key Results & Explainability (XAI) Findings

### Confirmed, Statistically Significant Findings (Welch's t-test, Bonferroni-corrected α = 0.01)

| PDE | Config | p-value | Finding |
|-----|--------|---------|---------|
| **Burgers'** | Q1 (3q, angle, basic) | **0.0092** ✅ | Significant Rel L2 improvement over C0 |
| **Navier–Stokes** | Q3 (5q, angle, basic) | **0.0037** ✅ | Strongest confirmed result — mean Rel L2 (u,v,p): Q3=0.524 vs C0=0.971 |

> ⚠️ Q3/Q5 single-seed advantage on Burgers' is suggestive but **not statistically confirmed** — Stage C sweep was runtime-interrupted. Heat equation has no multi-seed test. See `results/README.md` for caveats.

### Burgers' Equation — Master Performance Table

| Config | Rel L2 Error | PDE Loss | Params | Time (s) |
|--------|-------------|----------|--------|----------|
| C0 (classical) | 0.4247 | 0.03274 | 2,051 | 25.0 |
| Q1 (3q, angle, basic) | 0.3688 | 0.03021 | 2,283 | 426.1 |
| Q2 (4q, angle, basic) | 0.2790 | 0.02731 | 2,345 | 836.9 |
| Q3 (5q, angle, basic) | **0.2303** | 0.02692 | 2,407 | 1,897.9 |
| Q4 (3q, reupload) | 0.4392 | 0.04847 | 2,283 | 569.7 |
| Q5 (3q, angle, strong) | **0.1419** | 0.01033 | 2,307 | 366.4 |

### Heat Equation — Master Performance Table

| Config | Rel L2 Error | PDE Loss | Params | Time (s) |
|--------|-------------|----------|--------|----------|
| C0 | 0.02553 | 3.64e-4 | 2,051 | 24.9 |
| Q1 | 0.01927 | 3.88e-4 | 2,283 | 348.6 |
| Q2 | 0.02853 | 6.19e-5 | 2,345 | 751.1 |
| Q3 | 0.01059 | 1.88e-4 | 2,407 | 1,801.0 |
| Q4 | 0.02982 | 3.44e-4 | 2,283 | 494.5 |
| Q5 | **0.00623** | 2.21e-4 | 2,307 | 325.1 |
| C0-Best (5,081p, 5k ep) | **0.00386** | 1.37e-4 | 5,081 | 61.1 |

### Navier–Stokes — Master Performance Table

| Config | Rel L2 (u,v,p) | Rel L2 u | Rel L2 v | Rel L2 p | Params | Time (s) |
|--------|----------------|----------|----------|----------|--------|----------|
| C0 | 1.3203 | 0.0483 | 0.1573 | **4.7857** | 2,103 | 124.1 |
| Q1 | 0.5596 | 0.4935 | 1.1000 | 1.0385 | 2,332 | 293.4 |
| Q2 | 0.6352 | 0.5871 | 1.2578 | 1.0078 | 2,393 | 333.7 |
| Q3 | **0.4314** | 0.3302 | 1.3417 | 0.9906 | 2,454 | 400.2 |
| Q4 | 0.5296 | 0.4689 | 1.0193 | 0.9744 | 2,332 | 359.1 |
| Q5 | 0.5743 | 0.5089 | 1.0906 | 1.0553 | 2,350 | 379.5 |

### Fourier Spectrum — Circuit-Only Analysis

| Config | Burgers max freq | Heat max freq | NS max freq |
|--------|-----------------|---------------|-------------|
| Q1 (3q) | 0.635 (5 bins) | 0.635 (5 bins) | 0.635 (5 bins) |
| Q2 (4q) | 0.635 (5 bins) | 0.635 (5 bins) | 0.794 (6 bins) |
| Q3 (5q) | 0.953 (7 bins) | 0.635 (5 bins) | 0.794 (6 bins) |
| Q4 (reupload) | **1.112 (7–8 bins)** | **1.112 (7 bins)** | **1.112 (8 bins)** |
| Q5 (strong ent.) | 0.477 (4 bins) | 0.635 (5 bins) | 0.635 (5 bins) |

Q4's wider spectrum matches Schuld et al. [3] — reuploading multiplies accessible frequency components. **But wider expressivity ≠ better accuracy**: Q4 is never the best performer and shows elevated training-time quantum-gradient variance.

### McClean Gradient-Variance Scan (Barren-Plateau Diagnostic)

| Config | Burgers | Heat | NS |
|--------|---------|------|----|
| Q1 (3q) | 0.2586 | 0.2240 | 0.2464 |
| Q2 (4q) | **0.0344** | **0.0369** | **0.0287** |
| Q3 (5q) | 0.0345 | 0.0452 | 0.0321 |
| Q4 (reupload) | 0.0790 | 0.0765 | 0.0820 |
| Q5 (strong ent.) | 0.0836 | 0.0868 | 0.0601 |

**The single most reproducible mechanistic finding**: sharp ~7–8× gradient-variance drop from 3→4 qubits across all three PDEs using identical circuits. Directionally consistent with barren-plateau onset; the qubit range (3–5) is too narrow to confirm exponential collapse.

### Q5 Weight-Movement — PDE-Dependent Trainability

| PDE | Weight change after training |
|-----|------------------------------|
| Burgers | Small (near-frozen circuit) |
| Heat | 3.33% — flagged as "weights barely moved" |
| **Navier–Stokes** | **7.30%** — "weights moved meaningfully; quantum layer participated actively" |

Same fixed Q5 circuit, two qualitatively different training regimes purely as a function of PDE loss landscape — the most distinctive "how" finding of this project.

### Results Summary

The experimental study demonstrates that the effectiveness of Quantum-Assisted Physics-Informed Neural Networks (QAPINNs) is highly dependent on both the underlying partial differential equation and the chosen quantum circuit architecture. Rather than consistently outperforming classical PINNs, the introduction of a quantum layer alters the learning dynamics in ways that vary across different problem domains.

The key findings of this study include:

- The 5-qubit angle-encoded QAPINN (Q3) achieved the strongest statistically significant improvement over the classical PINN on the Navier–Stokes (Kovasznay Flow) benchmark.
- Data re-uploading consistently increased the accessible Fourier spectrum, indicating greater representational capacity, but did not consistently translate into improved predictive performance.
- Increasing the number of qubits reduced gradient variance, highlighting a trade-off between model expressivity and trainability.
- The strongly entangling ansatz exhibited PDE-dependent training behavior, demonstrating that the effectiveness of a quantum circuit depends on the characteristics of the underlying differential equation.
- Overall, the results indicate that quantum layers provide problem-specific advantages rather than serving as a universal replacement for classical PINNs.

---

## Design Methodology

From §7.1 of the technical report, a first-pass rule for choosing a QAPINN configuration:

1. **Start with the smallest circuit** (3 qubits, angle encoding, basic entangler) — most reliably trainable and the only config confirmed significant on the primary Burgers' benchmark.
2. **Only increase qubit count** if the classical baseline shows a specific, diagnosable fitting weakness — more qubits buys expressivity but costs gradient-variance headroom.
3. **Avoid strongly-entangling ansätze by default** — use the weight-movement check early; if weights stay near-frozen, switch to a simpler ansatz.
4. **Use data-reuploading selectively** for cases that specifically need wider frequency content, not as a general accuracy lever.

---

## Repository Structure

```
bqp_pinn_qapinn/
├── README.md                          ← This file
├── REPORT.pdf                         ← Technical report (PDF version)
├── PRESENTATION.pptx                  ← Project presentation
├── requirements.txt
├── LICENSE
│
├── models/
│   ├── pinn_classical.py              ← ClassicalPINN (FC baseline)
│   ├── qapinn.py                      ← QAPINN hybrid architecture
│   └── quantum_layer.py               ← JAX↔PyTorch differentiable bridge
│
├── notebooks/
│   ├── Burgers_Equation.ipynb         ← Primary benchmark (full outputs)
│   ├── Heat_Equation.ipynb            ← Linear PDE generalization check
│   └── NavierStokes_Kovasznay.ipynb   ← 2D NS stress test (full outputs)
│
├── results/
│   ├── README.md                      ← Results interpretation + caveats
│   ├── Burgers/
│   │   ├── metrics.csv                ← Per-config performance data
│   │   ├── training_loss.png
│   │   ├── comparison_bars.png
│   │   ├── fourier_spectrum.png
│   │   ├── activation_maps.png
│   │   └── master_performance_table.png
│   ├── Heat/
│   │   └── (same structure)
│   └── Navier_Stokes/
│       └── (same structure)
│
├── docs/
│   ├── methodology.md                 ← Mathematical framework & derivations
│   ├── mathematical_analysis.md       ← Fourier + barren-plateau derivations
│   └── reproducibility.md            ← Environment setup & hyperparameters
│
└── references/
    ├── README.md
    └── bibtex_references.bib
```

---

## Setup & Reproducibility

### Installation

Clone the repository:
```bash
git clone https://github.com/Harshitha465/bqp_pinn_qapinn.git
cd bqp_pinn_qapinn
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Key pinned dependencies:**
- `torch >= 2.1.0` (classical layers, CUDA)
- `pennylane == 0.45.1` (quantum circuits)
- `jax == 0.9.2` + `jaxlib == 0.9.2` (pinned — PennyLane's JAX interface uses `jax.core.is_concrete`, removed in JAX 0.10+)

**Hardware used:** Google Colab, Tesla T4 GPU (15.64 GB VRAM), JAX 0.7.2 (GPU-visible). Classical layers run on CUDA; quantum circuits run on CPU via PennyLane's `default.qubit` simulator (JAX interface, JIT-compiled).

**Global seed:** `SEED = 42` for all single-seed main-matrix runs. Stage C sweeps use seeds `[11, 29, 47, 97, ...]`.

Run notebooks in order:
1. `notebooks/Heat_Equation.ipynb`
2. `notebooks/Burgers_Equation.ipynb`
3. `notebooks/NavierStokes_Kovasznay.ipynb`

See [`docs/reproducibility.md`](docs/reproducibility.md) for full hyperparameter table and environment details.

---

## Future Work

- Extend the evaluation to more complex and higher-dimensional PDEs, including full Navier–Stokes flow problems, to further assess the scalability and robustness of the proposed approach.
- Incorporate **Quantum Reservoir Computing Frameworks (QRCF)** as an additional baseline for comparison alongside Classical PINNs and QAPINNs.
- Explore alternative quantum data encoding methods, variational circuit architectures, and optimization strategies to improve model accuracy, convergence, and training stability.
- Evaluate the models on real quantum hardware and noise-aware simulators to study practical deployment and robustness.
- Investigate FPGA-based hardware acceleration for the classical components of hybrid quantum-classical PINN training to improve computational efficiency, scalability, and energy efficiency.
- Develop more efficient hybrid quantum-classical workflows for solving large-scale scientific and engineering problems.

---

## Team Contributions

This project was developed collaboratively by **Lingerker Harshita Bai**, **Ansh Goel**, and **Amavasya Manoj Kumar Reddy**. Each team member independently implemented and experimented on all three benchmark problems—**Burgers' Equation, Heat Equation, and Navier–Stokes (Kovasznay Flow)**. Individual implementations were systematically evaluated, compared, and reviewed through collaborative discussions.

The final notebooks and repository represent a unified implementation that integrates the strongest ideas, validated methodologies, and best-performing approaches from each member's work, resulting in a consistent, reproducible, and optimized solution across all benchmark problems.

### Primary Areas of Contribution

| Team Member | Primary Contribution |
|-------------|----------------------|
| **Lingerker Harshita Bai** | Led the implementation of the Classical PINN and QAPINN models, conducted experiments and performance analysis across all benchmark PDEs, integrated the final notebooks, organized the project repository, and documented the experimental findings. |
| **Ansh Goel** | Focused on designing and implementing the experimental framework, evaluating multiple quantum circuit configurations, conducting systematic benchmarking, analyzing parameter efficiency and training performance, and deriving evidence-based conclusions through comprehensive experimental studies. |
| **Amavasya Manoj Kumar Reddy** | Led the research methodology and project design, developed the theoretical and mathematical foundation, designed the experimental investigation, conducted literature review and scientific analysis, interpreted the experimental findings, and prepared the technical report and supporting project documentation. |

---

## Limitations

> Full discussion in the Technical Report §6 and [`results/README.md`](results/README.md).

- **Burgers' Stage C is incomplete** — Q2/Q3/Q5 multi-seed sweep was runtime-interrupted (Colab session limit). Only Q1 and Q4 have valid 5-seed statistics. Q3/Q5 single-seed wins are suggestive, not confirmed.
- **Heat equation has no multi-seed test** — all heat numbers are single-seed (SEED=42). The `RUN_STAGE_C=True` flag exists but no Stage C execution cell was written in that notebook.
- **NS Stage C covers C0 and Q3 only** — Q3-vs-C0 (p=0.0037) is real; other NS configs rest on single-seed data.
- **NS cross-seed McClean robustness reports std=0.000** — seed was not re-applied per run in that specific cell; treat NS McClean numbers as single-seed only.
- **Only expectation-value measurement** (`⟨Z⟩` per qubit) tested — probability-vector measurement not evaluated (noted scope gap per WISER brief).

---

## References

The research papers and supporting resources used in this project are available in the `references/` folder.

[1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. *Physics-Informed Neural Networks.* Journal of Computational Physics, 2019.

[2] Shah, N., Lineswala, P., & Chopra, A. *Benchmarking QA-PINN for CFD.* IEEE Quantum Week (QCE), 2024.

[3] Schuld, M., Sweke, R., & Meyer, J. J. *Effect of data encoding on the expressive power of variational quantum ML models.* Physical Review A, 103, 032430 (2021).

[4] McClean, J. R., Boixo, S., Neven, H., & Babbush, R. *Barren plateaus in quantum neural network training landscapes.* Nature Communications, 9, 4812 (2018).

---

## Acknowledgements

This project was carried out as part of the **WISER Quantum + AI Summer Program 2026** and submitted for the **BQP Challenge 2026**. We acknowledge the guidance, learning opportunities, and platform provided through the program.

## License

This project is intended for academic and research purposes.

*BQP WISER Global Quantum+AI 2026 — Full technical report: [`REPORT.pdf`](REPORT.pdf) | References: [`references/`](references/)*
