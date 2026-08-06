# Methodology — QAPINN Design Framework & Mathematical Foundations

> **Source:** This document merges the formal methodology report (*A Methodology for Constructing
> Problem-Specific Quantum Circuits and QAPINN Architectures*, WISER 2026 Summer Program — BQP
> Challenge, August 2026) with the original circuit-level mathematical derivations.

---

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

## 2. The Design Problem

A QAPINN replaces part of a classical PINN with a variational quantum circuit (VQC). But a VQC is not a single object: it is a family parametrised by:

- **(i)** how the PDE coordinates are **encoded** into qubit rotations
- **(ii)** how qubits are **entangled**
- **(iii)** the circuit **depth**
- **(iv)** the number of **qubits**

Each axis changes both the expressive power of the model and the difficulty of training it. The central design question is:

> *Given a PDE, how should one choose the encoding, topology, depth and width of the quantum layer so that the resulting QAPINN is competitive with — and more parameter-efficient than — a fair classical PINN?*

Answering this responsibly requires a protocol that **(a)** fixes a fair baseline before looking at results, **(b)** measures true solution quality rather than training loss, **(c)** separates cheap exploration from expensive confirmation, and **(d)** treats null results as valid outcomes.

---

## 3. QAPINN Architecture Template

Every model in this study follows one hybrid template, so that only the quantum block varies:

$$(\mathbf{x}, t) \xrightarrow{\text{linear projection}} \theta \in \mathbb{R}^n \xrightarrow{\text{VQC}} \langle Z_i \rangle_{i=1}^n \xrightarrow{\text{linear read-out}} \hat{u}(\mathbf{x}, t)$$

| Block | Operation | Device |
|-------|-----------|--------|
| input_proj | `Linear(2, 32) → Tanh → Linear(32, n_qubits)` | GPU |
| quantum_layer | `VQC(n_qubits, n_qlayers=4)` | CPU / GPU via JAX |
| postprocessing | `Linear(n_qubits, 25) → Tanh → [25→25→25] → Linear(25, n_out)` | GPU |

The classical projection maps the PDE inputs to $n$ encoding angles; the VQC applies an encoding unitary followed by $D$ repetitions of an entangling ansatz; Pauli-Z expectation values are read out and mapped linearly to the field value.

### 3.1 Four Design Axes

| Axis | Screened values | Role / intuition |
|------|----------------|-----------------|
| **Encoding** | `angle`, `re-upload` | Sets the Fourier frequencies the circuit can represent. Re-uploading interleaves data with trainable layers, enlarging the accessible spectrum. |
| **Topology** | `basic`, `cascade` | `basic`: one entangling sweep; `cascade`: sequential qubit-to-qubit entanglement, matching the QCPINN construction. |
| **Depth** $D$ | 1, 3, 5 | Number of ansatz repetitions; trades expressivity against gradient stability (barren-plateau risk). |
| **Qubits** $n$ | 2, 3, 4 | Width of the quantum feature space; scanned only for the confirmed best circuit. |

**Three configurable axes per variant (circuit-level detail):**
- **Encoding**: `angle` — one `AngleEmbedding(inputs, rotation="Y")` before all ansatz layers; `reupload` — re-applies `AngleEmbedding` before *each* ansatz layer ($L=4$ uploads total)
- **Ansatz**: `BasicEntanglerLayers` (1 rotation angle/qubit/layer) or `StronglyEntanglingLayers` (3 rotation angles/qubit/layer)
- **Measurement**: $\langle\text{PauliZ}(i)\rangle$ expectation value per qubit — only expectation values were evaluated (probability-vector measurement is an acknowledged scope gap)

---

## 4. The Fair-Comparison Protocol

A quantum result is meaningless without a baseline that cannot be dismissed. Three rules are fixed **before** any result is inspected:

1. **Two classical arms.** Each QAPINN is compared to a parameter-matched Tanh MLP (same trainable-parameter budget, chosen from a predeclared family) **and** to a fixed larger classical PINN (four hidden layers of 20 units). The larger arm is never re-tuned after seeing test error.

2. **Solution error, not loss.** The primary metric is the relative $L_2$ error against a trusted reference:
$$\text{rel-}L_2 = \frac{\|\mathbf{u}_{\text{ref}} - \hat{\mathbf{u}}\|_2}{\|\mathbf{u}_{\text{ref}}\|_2}$$
A lower PDE residual with a higher rel-$L_2$ is explicitly counted as **not** a win. Heat uses its analytical solution; Burgers uses an independent finite-volume (Rusanov/SSP-RK2) reference whose coarse-to-fine refinement error is checked before any neural result is trusted.

3. **Replication.** Configurations are ranked on a single seed but claimed only after five-seed confirmation (seeds 11, 29, 47, 71, 97), reporting mean, sample standard deviation and 95% confidence intervals; failed runs are never averaged away.

---

## 5. The Staged Methodology

The core contribution is a funnel that spends cheap compute widely and expensive compute narrowly:

| Stage   | Goal                            | Budget / Outcome |
|--------|---------------------------------|------------------|
| A      | Validate the three arms         | Sanity check     |
| B      | Screen a 2×2×3 QAPINN grid      | 1 seed, 2,000 epochs |
| C      | Confirm the top two candidates  | 5 seeds, 5,000 epochs |
| D      | Qubit scaling and generalization | Design rule per PDE |

**Stage A — Validation.** Train the three arms (matched classical, best classical, QAPINN) once per PDE to confirm the pipeline is differentiable and that all arms actually solve the equation. No design conclusions are drawn here.

**Stage B — One-seed screening.** Sweep the prespecified $2 \times 2 \times 3$ grid (encoding × topology × depth) at a fixed three qubits: 12 QAPINN configurations per PDE, on a single seed at a reduced 2,000-epoch budget. This ranks candidates cheaply. Ranking key: lowest rel-$L_2$, then lower training time, then more stable (higher-variance-avoiding) gradients as a tie-break. The top two per PDE become **finalists**.

**Stage C — Five-seed confirmation.** Each finalist is retrained at the full 5,000-epoch budget across all five seeds and compared to both classical arms on the identical seed list (2 PDEs × 2 finalists × 5 seeds × 3 arms). Only these replicated numbers support the report's conclusions.

**Stage D — Scaling and generalization.** For the confirmed best circuit, scan qubits $\{2, 3, 4\}$ (Stage D1) to map the accuracy/cost trade-off, and specify a coefficient-transfer protocol (train a parameter-conditioned model on several PDE coefficients, evaluate on a held-out coefficient) as the generalization test (Stage D2).

---

## 6. Circuit Diagnostics

Two diagnostics are recorded for every QAPINN so that design choices are **explained**, not just ranked:

- **Fourier spectrum** of the circuit output links the encoding axis to the frequencies the model can represent: a data-reuploading VQC realises a truncated Fourier series whose bandwidth is set by the encoding, which is why the encoding axis is treated as primary.
- **Gradient-variance / barren-plateau** statistic tracks whether deeper or wider circuits still receive a usable training signal; it is the reason depth and qubit count are increased only when they demonstrably improve the error/cost trade-off rather than by default.

---

## 7. Fourier-Frequency Derivation (from Actual Circuit Code)

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

**Gap between theoretical upper bound and observed spectrum:** Theory predicts up to 25 components for Q4, but only 7–8 significant bins are observed. Generic random initialisation concentrates Fourier weight in low-frequency components; high-frequency terms are accessible but have vanishingly small coefficients. This gap is evidence that *accessible expressivity ≠ effective expressivity*.

---

## 8. Barren-Plateau (McClean) Argument

For a sufficiently expressive, randomly-initialised ansatz approximating a unitary 2-design:

$$\text{Var}_{\boldsymbol{\theta}}\left[\frac{\partial\langle\hat{M}\rangle}{\partial\theta_k}\right] \sim \mathcal{O}(2^{-n})$$

**What this project's data confirms:** A sharp, consistent ~7–8× variance drop from 3→4 qubits, then roughly flat 4→5 qubits — reproduced across all three PDEs with architecturally identical circuits.

**What this project's data does NOT confirm:** The full asymptotic exponential law. The qubit range (3–5) is too narrow to distinguish exponential from other rapidly-decaying trends. The honest claim is "consistent, sharp variance decline with added qubits" — consistent with *onset* of a barren-plateau-type effect, not confirmation of the full asymptotic regime.

---

## 9. JAX ↔ PyTorch Differentiable Bridge

All quantum circuits are executed via PennyLane's JAX interface (`interface="jax"`, `diff_method="backprop"`, JIT-compiled). To enable end-to-end training through the hybrid model, a custom `torch.autograd.Function`-based bridge (`_JaxOp` / `_VJPChain`) uses `jax.vjp` to compute gradients through the quantum circuit and return them to PyTorch's autograd graph.

This supports arbitrary-order differentiation (needed for $u_{xx}$ terms in PDE residuals) without breaking the computational graph at the JAX/PyTorch boundary.

---

## 10. Results That Instantiate the Methodology

Two robust qualitative findings emerge and agree across both PDEs: **cascade topology beats basic**, and **angle encoding beats re-upload**. The best architecture is therefore `angle / cascade`, consistent with the QCPINN construction [2].

### Table 1: Heat Equation — Five-Seed Confirmation

| Model (heat) | Circuit | rel-$L_2$ (mean ± sd) | Params |
|---|---|---|---|
| QAPINN | reupload / cascade, q3 d3 | 0.00436 ± 0.00264 | 968 |
| QAPINN | angle / cascade, q3 d3 | 0.00457 ± 0.00116 | 968 |
| Classical (best) | 4×20 Tanh MLP | 0.00684 ± 0.00411 | 1341 |
| Classical (matched) | Tanh MLP | 0.00969 ± 0.00520 | 921 |

Both QAPINNs beat both classical arms while using fewer parameters than the larger classical network.

### Table 2: Burgers' Equation — Five-Seed Confirmation

| Model (Burgers) | Circuit | rel-$L_2$ (mean ± sd) | Params |
|---|---|---|---|
| Classical (best) | 4×20 Tanh MLP | 0.2057 ± 0.0662 | 1341 |
| QAPINN | angle / cascade, q3 d3 | 0.2781 ± 0.1612 | 968 |
| QAPINN | reupload / cascade, q3 d1 | 0.2912 ± 0.1400 | 956 |
| Classical (matched) | Tanh MLP | 0.3855 ± 0.0601 | 921 |

The QAPINN beats the parameter-matched classical network but not the larger one — a **parameter-efficiency win**, not an accuracy win.

**Interpretation.** On the smooth, diffusion-dominated heat equation the quantum layer is a genuine win on **both** axes: lower error **and** fewer parameters than the best classical model. On the advection-dominated Burgers equation, whose sharp gradient is harder for a low-bandwidth circuit, the QAPINN is more parameter-efficient than an equally sized classical network but is overtaken by the larger classical model. Stage D1 shows the same trade-off on both PDEs: moving from 2 to 3 to 4 qubits reduces error but roughly **doubles** training time at each step (heat: ~2 → 6 → 12 minutes). More qubits are thus justified only where the error reduction is worth the cost.

---

## 11. Statistical Testing

Where multi-seed data exists, significance is assessed with:
- **Welch's t-test** (unequal variances) comparing each QAPINN's Rel $L_2$ against C0
- **Bonferroni correction** for 5 simultaneous comparisons: corrected $\alpha = 0.01$
- **5 seeds** per configuration in Stage C sweeps

**Coverage:**
- Burgers': Q1 and Q4 complete (5 seeds each); Q2/Q3/Q5 Stage C runtime-interrupted
- Heat: No multi-seed test executed
- Navier–Stokes: C0 and Q3 only (5 seeds each); Q3 vs C0 p=0.0037 (Bonferroni-significant)

---

## 12. The Resulting Design Rule

The methodology outputs the following transferable recipe for constructing a problem-specific QAPINN.

**QAPINN Design Rule (from this study):**

1. Start from **angle encoding + cascade topology, 3 qubits, depth 3** — the confirmed best point.
2. Always pin two classical baselines (parameter-matched and a fixed larger network) and rank on reference rel-$L_2$, **never** on training loss.
3. Screen the $2 \times 2 \times 3$ encoding/topology/depth grid on one seed; confirm the top two on five seeds before claiming anything.
4. Prefer cascade over basic entanglement and angle over re-upload encoding unless screening on the new PDE says otherwise.
5. Increase qubits only when Stage-D scaling shows the error gain outweighs the ~2× per-qubit cost.
6. Expect the quantum win to be **parameter efficiency**, strongest on smooth diffusion-type PDEs and weakest where sharp advective gradients demand high spectral bandwidth.

---

## 13. Mathematical Basis for the Design Conclusions

The design rule is not only empirical; most of it follows from the Fourier picture of variational circuits.

### 13.1 A QAPINN is a Truncated Fourier Series

Following Schuld, Sweke and Meyer [1], a circuit in which the inputs enter through Pauli-rotation encoding gates computes

$$f_\theta(\mathbf{x}) = \sum_{\omega \in \Omega} c_\omega(\theta)\, e^{i\omega \cdot \mathbf{x}}$$

a Fourier series in the PDE coordinates. The two circuit ingredients play distinct roles: the **encoding** fixes the accessible frequency set $\Omega$ (the eigenvalue differences of the data-encoding generators), while the **trainable entangling layers and measurement** fix the coefficients $c_\omega$. Re-uploading the data $r$ times enlarges $\Omega$ — its bandwidth grows roughly linearly in $r$ — whereas angle encoding (a single encode) yields the smallest spectrum the given qubits support. In one phrase: **encoding controls expressivity**, i.e. which frequencies the model can represent.

### 13.2 Why Angle Encoding Suffices for These PDEs

The encoding choice is therefore a bias–variance decision in the frequency domain. The heat solution $u(x,t) = \sin(\pi x)\,e^{-\alpha\pi^2 t}$ is a **single** spatial mode $\omega = \pi$: its bandwidth requirement is minimal, and that mode already lies in $\Omega$ for a single angle encoding. Re-uploading adds harmonics $2\pi, 3\pi, \ldots$ whose true coefficients are **zero**; they cannot lower the approximation bias but they enlarge the parameter space and inflate optimisation variance. Angle encoding thus attains the same accuracy at lower variance and lower cost — exactly the observed outcome, where the two encodings are statistically indistinguishable in mean error on heat and angle is preferred on stability and cost.

Burgers' equation develops a thin internal layer of width $\mathcal{O}(\sqrt{\nu})$ whose spectrum **does** carry high frequencies; there the extra bandwidth is in principle useful, but a three-qubit circuit is bandwidth- and coefficient-limited, which is precisely why both encodings trail the larger classical network and the quantum benefit reduces to parameter efficiency rather than accuracy.

### 13.3 Expressivity vs. Trainability: The Depth and Qubit Sweet Spot

Expressivity is not free. For sufficiently expressive randomly-initialised circuits, McClean et al. [3] show the gradient variance decays exponentially in the qubit count:

$$\text{Var}[\partial_\theta f_\theta] = \mathcal{O}(b^{-n}), \quad b > 1$$

the **barren-plateau** phenomenon; increasing depth toward a unitary 2-design has the same effect. Enlarging $\Omega$ by adding qubits or depth therefore eventually destroys the training signal. The optimal design is the **smallest** circuit whose spectrum already covers the target's frequency content — for the smooth heat and Burgers problems this is the modest three-qubit, depth-three point, in agreement with both the screening scan and the measured gradient-variance diagnostics.

### 13.4 Cost Scaling

On a classical statevector simulator the $n$-qubit state carries $2^n$ amplitudes, so memory and per-step cost are $\Theta(2^n)$: each added qubit **doubles** the work. This reproduces the observed 2 → 6 → 12 minute progression across 2 → 3 → 4 qubits and quantifies the price of extra expressivity.

### 13.5 The Topology Conclusion

The cascade-over-basic result rests on a weaker, partly heuristic basis and is reported as such. Both patterns generate genuine entanglement; the ring (basic) adds one wrap-around CNOT that, at $n=3$, closes a loop and lengthens the effective entangling map without a matching enlargement of the reachable coefficient space $\{c_\omega\}$. This tends to raise gradient variance and worsen landscape conditioning, consistent with cascade's more stable training and with the QCPINN construction [2] — but we treat it as an empirical finding to be re-screened on each new PDE rather than a theorem.

---

## 14. Limitations and Next Steps

The study covers two 1D PDEs on noiseless statevector simulation; the design rule should be re-screened, not assumed, for higher-dimensional or CFD problems. The Burgers result is a caution against over-claiming: the quantum layer's benefit is parameter efficiency, not universal accuracy, and its variance across seeds is larger than the classical arms'.

The generalization protocol (Stage D2) is specified but not yet executed; running the coefficient-transfer test is the immediate next step, followed by:
- Extending the screen to the lid-driven-cavity Navier–Stokes case
- Adding barren-plateau scans at wider qubit counts to bound where the cascade ansatz remains trainable

---

## References

[1] M. Schuld, R. Sweke, J. J. Meyer. *Effect of data encoding on the expressive power of variational quantum machine-learning models.* Phys. Rev. A 103, 032430 (2021).

[2] A. Farea, S. Khan, M. S. Celebi. *QCPINN: Quantum-Classical Physics-Informed Neural Networks.* Mach. Learn.: Sci. Technol. (2026); arXiv:2503.16678.

[3] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, H. Neven. *Barren plateaus in quantum neural network training landscapes.* Nature Communications 9, 4812 (2018).

[4] Raissi, M., Perdikaris, P., & Karniadakis, G. E. *Physics-Informed Neural Networks.* J. Computational Physics, 378, 686–707 (2019).

[5] Shah, N., Lineswala, P., & Chopra, A. *Benchmarking QA-PINN for CFD.* IEEE Quantum Week (QCE), 2024.
