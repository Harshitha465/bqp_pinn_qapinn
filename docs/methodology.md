# Methodology — Mathematical Framework & Circuit Derivations

## 1. Problem Formulation

### 1.1 Physics-Informed Neural Network (PINN) Loss

A PINN parametrises the PDE solution $\hat{u}_\theta(x, t) = \text{NN}(x, t; \theta)$ and minimises:

$$\mathcal{L}(\theta) = w_{ic}\,\mathcal{L}_{ic}(\theta) + w_{bc}\,\mathcal{L}_{bc}(\theta) + w_{r}\,\mathcal{L}_{r}(\theta)$$

where each term is the mean-squared residual over sampled collocation, initial, and boundary points computed via automatic differentiation. For all experiments: $w_{ic} = w_{bc} = w_r = 1.0$.

### 1.2 PDE Benchmarks

**1D Heat Equation (linear diffusion):**
$$u_t = \alpha\,u_{xx}, \quad x \in [-1,1],\; t \in [0,1], \quad \alpha = 0.1$$
$$u(x,0) = \sin(\pi x), \quad u(\pm 1, t) = 0 \qquad \text{Exact: } u = \sin(\pi x)\,e^{-\alpha\pi^2 t}$$

**1D Viscous Burgers' Equation (nonlinear, shockwave):**
$$u_t + u\,u_x = \nu\,u_{xx}, \quad \nu = 0.01/\pi, \quad u(x,0) = -\sin(\pi x), \quad u(\pm 1, t) = 0$$
Ground truth via Cole–Hopf transform + Gauss–Hermite quadrature.

**2D Kovasznay Flow (steady incompressible Navier–Stokes):**
$$u\,u_x + v\,u_y = -p_x + \nu(u_{xx}+u_{yy}), \quad u\,v_x + v\,v_y = -p_y + \nu(v_{xx}+v_{yy}), \quad u_x + v_y = 0$$
$$\nu = 1/Re,\quad Re = 40, \quad \lambda = \frac{Re}{2} - \sqrt{\frac{Re^2}{4} + 4\pi^2} \approx -0.9637$$
$$\text{Exact: } u = 1 - e^{\lambda x}\cos(2\pi y),\quad v = \frac{\lambda}{2\pi}e^{\lambda x}\sin(2\pi y),\quad p = \frac{1}{2}(1 - e^{2\lambda x})$$

---

## 2. QAPINN Architecture

The QAPINN replaces the first hidden layer of a classical PINN with a VQC:

```
input_proj: Linear(2, 32) → Tanh → Linear(32, n_qubits)   [GPU]
    ↓
quantum_layer: VQC (n_qubits, n_qlayers=4)                 [CPU/GPU via JAX]
    ↓
postprocessing: Linear(n_qubits, 25) → Tanh → [25→25→25] → Linear(25, n_out)  [GPU]
```

**Three configurable axes per variant:**
- **Encoding**: `angle` — one `AngleEmbedding(inputs, rotation="Y")` before all ansatz layers; `reupload` — re-applies `AngleEmbedding` before *each* ansatz layer (L=4 uploads total)
- **Ansatz**: `BasicEntanglerLayers` (1 rotation angle/qubit/layer) or `StronglyEntanglingLayers` (3 rotation angles/qubit/layer)
- **Measurement**: `⟨PauliZ(i)⟩` expectation value per qubit — only expectation values were evaluated (probability-vector measurement is an acknowledged scope gap)

---

## 3. Fourier-Frequency Derivation (from Actual Circuit Code)

Per Schuld et al. [3], the encoding gate type determines the accessible frequency spectrum. For this project's `AngleEmbedding(inputs, rotation="Y")`:

- Each qubit receives `RY(x_i)` with generator $Y/2$, eigenvalues $\pm 1/2$
- Frequency contribution per qubit: $\omega_i \in \{-1, 0, +1\}$ (pairwise differences of eigenvalues $\times 2$)

**For `angle` encoding (1 upload, $n$ qubits):**
$$\Omega = \{k = \sum_{i=1}^{n} \omega_i : \omega_i \in \{-1,0,+1\}\} \implies k \in [-n, +n]$$
Up to $2n+1$ distinct frequency components.

**For `reupload` encoding ($L=4$ uploads, $n=3$ qubits):**
$$\Omega_{\text{reupload}} \subseteq \{k \in [-Ln, +Ln]\} \implies k \in [-12, +12]$$
Upper bound: $2 \cdot L \cdot n + 1 = 25$ components (vs. Q1's $7$).

**Key confirmed prediction:** Since the *encoding gate* sets $\Omega$ (not the ansatz), switching `BasicEntanglerLayers → StronglyEntanglingLayers` (Q1 → Q5, same `angle` encoding) should **not** change the accessible frequency range. Measured data confirms this — Q1 (max freq 0.635, 5 bins) and Q5 (0.477, 4 bins) are close across all three PDEs; Q4 (reupload) consistently dominates (1.112, 7–8 bins).

**Gap between theoretical upper bound and observed spectrum:** Theory predicts up to 25 components for Q4, but only 7–8 significant bins are observed. Generic random initialisation concentrates Fourier weight in low-frequency components; high-frequency terms are accessible but have vanishingly small coefficients. This gap is evidence that accessible expressivity ≠ effective expressivity.

---

## 4. Barren-Plateau (McClean) Argument

For a sufficiently expressive, randomly-initialised ansatz approximating a unitary 2-design:

$$\text{Var}_{\boldsymbol{\theta}}\left[\frac{\partial\langle\hat{M}\rangle}{\partial\theta_k}\right] \sim \mathcal{O}(2^{-n})$$

**What this project's data confirms:** A sharp, consistent ~7–8× variance drop from 3→4 qubits, then roughly flat 4→5 qubits — reproduced across all three PDEs with architecturally identical circuits.

**What this project's data does NOT confirm:** The full asymptotic exponential law. The qubit range (3–5) is too narrow to distinguish exponential from other rapidly-decaying trends. The honest claim is "consistent, sharp variance decline with added qubits" — consistent with *onset* of a barren-plateau-type effect, not confirmation of the full asymptotic regime.

---

## 5. JAX ↔ PyTorch Differentiable Bridge

All quantum circuits are executed via PennyLane's JAX interface (`interface="jax"`, `diff_method="backprop"`, JIT-compiled). To enable end-to-end training through the hybrid model, a custom `torch.autograd.Function`-based bridge (`_JaxOp` / `_VJPChain`) uses `jax.vjp` to compute gradients through the quantum circuit and return them to PyTorch's autograd graph.

This supports arbitrary-order differentiation (needed for `u_xx` terms in PDE residuals) without breaking the computational graph at the JAX/PyTorch boundary.

---

## 6. Statistical Testing

Where multi-seed data exists, significance is assessed with:
- **Welch's t-test** (unequal variances) comparing each QAPINN's Rel L2 against C0
- **Bonferroni correction** for 5 simultaneous comparisons: corrected $\alpha = 0.01$
- **5 seeds** per configuration in Stage C sweeps

**Coverage:**
- Burgers': Q1 and Q4 complete (5 seeds each); Q2/Q3/Q5 Stage C runtime-interrupted
- Heat: No multi-seed test executed
- Navier–Stokes: C0 and Q3 only (5 seeds each); Q3 vs C0 p=0.0037 (Bonferroni-significant)

---

## References

[1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. *Physics-Informed Neural Networks.* J. Computational Physics, 378, 686–707 (2019).

[2] Shah, N., Lineswala, P., & Chopra, A. *Benchmarking QA-PINN for CFD.* IEEE Quantum Week (QCE), 2024.

[3] Schuld, M., Sweke, R., & Meyer, J. J. *Effect of data encoding on the expressive power of variational quantum ML models.* Physical Review A, 103, 032430 (2021).

[4] McClean, J. R., Boixo, S., Neven, H., & Babbush, R. *Barren plateaus in quantum neural network training landscapes.* Nature Communications, 9, 4812 (2018).
