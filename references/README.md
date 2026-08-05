# References & Academic Citations

This directory contains bibliographic references supporting the BQP WISER Global Quantum+AI 2026 research repository on Quantum-Assisted Physics-Informed Neural Networks (QAPINN).

---

## Core References (directly cited in the Technical Report)

**[1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)**
*Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*
Journal of Computational Physics, 378, 686–707.
→ Foundational PINN formulation: PDE residual loss, automatic differentiation, BC/IC embedding.

**[2] Shah, N., Lineswala, P., & Chopra, A. (2024)**
*Benchmarking QA-PINN for CFD.*
IEEE International Conference on Quantum Computing and Engineering (QCE), 2024.
→ Direct motivation for this project's hybrid QAPINN architecture and CFD benchmark selection.

**[3] Schuld, M., Sweke, R., & Meyer, J. J. (2021)**
*Effect of data encoding on the expressive power of variational quantum machine learning models.*
Physical Review A, 103(3), 032430.
→ Fourier-series characterisation of VQC outputs; theoretical basis for §2.1/§5.1 of the Technical Report. Predicts that data-reuploading widens the accessible frequency spectrum — **confirmed across all three PDEs**.

**[4] McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018)**
*Barren plateaus in quantum neural network training landscapes.*
Nature Communications, 9(1), 4812.
→ Exponential gradient-variance collapse with qubit count for 2-design-like circuits; basis for §5.2 barren-plateau diagnostic. Project finds *onset* of this effect (sharp 3→4 qubit variance drop), not the full asymptotic regime.

---

## Supporting References

**[5] Kovasznay, L. S. G. (1948)**
*Laminar flow behind two-dimensional grid.*
Proceedings of the Cambridge Philosophical Society, 44(1), 58–62.
→ Closed-form analytical solution for 2D steady Navier–Stokes used as the NS benchmark ground truth.

**[6] Kyriienko, O., Paine, A. E., & Elfving, V. E. (2021)**
*Solving nonlinear differential equations with differentiable quantum circuits.*
Physical Review A, 103(5), 052416.
→ VQC-based PDE solvers using differentiable quantum circuits.

**[7] Hu, Z., Jagtap, A. D., Karniadakis, G. E., & Kawaguchi, K. (2022)**
*When do extended physics-informed neural networks (XPINNs) improve generalization?*
SIAM Journal on Scientific Computing.
→ Extended PINN architectures and generalisation analysis.

**[8] Cerezo, M., et al. (2021)**
*Variational quantum algorithms.*
Nature Reviews Physics, 3, 625–644.
→ Overview of VQA landscape, trainability, and circuit design trade-offs.

**[9] Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K. (2018)**
*Quantum circuit learning.*
Physical Review A, 98(3), 032309.
→ Quantum circuit as a machine learning model; parameter-shift gradient rule.

---

## BibTeX

See [`bibtex_references.bib`](bibtex_references.bib) for full BibTeX entries for all references above.
