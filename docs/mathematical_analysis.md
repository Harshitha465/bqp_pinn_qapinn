# Mathematical Derivations & Theoretical Analysis

## 1. Differential Equation Formulations

### 1.1 1D Heat Equation (Linear Diffusion)

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in [-1,1],\; t \in [0,1], \quad \alpha = 0.1
$$

Initial and boundary conditions:

$$
u(x, 0) = \sin(\pi x), \qquad u(\pm 1, t) = 0
$$

Exact solution (separation of variables, verified with SymPy):

$$
u(x, t) = \sin(\pi x)\,e^{-\alpha \pi^2 t}
$$

### 1.2 1D Viscous Burgers' Equation (Nonlinear Convective-Diffusive Transport)

$$
\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} = \nu\,\frac{\partial^2 u}{\partial x^2}, \quad \nu = \frac{0.01}{\pi}, \quad x \in [-1,1],\; t \in [0,1]
$$

Initial and Dirichlet boundary conditions:

$$
u(x, 0) = -\sin(\pi x), \qquad u(\pm 1, t) = 0
$$

Exact solution: Cole–Hopf transform reduces this to the heat equation; ground truth is computed via Gauss–Hermite quadrature (verified symbolically with SymPy).

### 1.3 2D Kovasznay Flow (Steady Incompressible Navier–Stokes)

$$
\begin{aligned}
u\,u_x + v\,u_y &= -p_x + \nu\,(u_{xx}+u_{yy}) \\
u\,v_x + v\,v_y &= -p_y + \nu\,(v_{xx}+v_{yy}) \\
u_x + v_y &= 0 \qquad (\text{continuity})
\end{aligned}
$$

where $\nu = 1/Re$, $Re = 40$. The Kovasznay parameter:

$$
\lambda = \frac{Re}{2} - \sqrt{\frac{Re^2}{4} + 4\pi^2} \approx -0.9637
$$

Exact analytical fields (closed-form, verified with SymPy):

$$
\begin{aligned}
u(x,y) &= 1 - e^{\lambda x}\cos(2\pi y) \\
v(x,y) &= \frac{\lambda}{2\pi}\,e^{\lambda x}\sin(2\pi y) \\
p(x,y) &= \frac{1}{2}\left(1 - e^{2\lambda x}\right)
\end{aligned}
$$

---

## 2. Fourier-Frequency Analysis — Derived from Actual Circuit Code

Per Schuld, Sweke, and Meyer [3], the output of a VQC with angle-encoded inputs is a truncated Fourier series:

$$
f(\mathbf{x}, \boldsymbol{\theta}) = \langle \hat{M} \rangle = \sum_{\omega \in \Omega} c_\omega(\boldsymbol{\theta})\,e^{i\omega x}
$$

The accessible frequency set $\Omega$ is determined entirely by the **encoding gate** (not the ansatz).

### 2.1 This Project's Exact Encoding

Every QAPINN variant uses `AngleEmbedding(inputs, rotation="Y")` — i.e., $RY(x_i)$ on each qubit $i$. The generator is $Y/2$ with eigenvalues $\pm 1/2$; the resulting per-qubit frequency contribution is:

$$
\omega_i \in \{-1, 0, +1\}
$$

(pairwise differences of eigenvalues, scaled by 2).

### 2.2 Angle Encoding (Q1, Q2, Q3, Q5) — One Upload

Each input coordinate independently encoded into $n$ qubits. Total accessible frequency set:

$$
\Omega = \left\{k = \sum_{i=1}^{n}\omega_i \;:\; \omega_i \in \{-1,0,+1\}\right\}, \quad k \in [-n, +n]
$$

Upper bound: $2n+1$ distinct components.

| Config | Qubits | Theoretical max freq components |
|--------|--------|----------------------------------|
| Q1, Q5 | 3 | 7 |
| Q2 | 4 | 9 |
| Q3 | 5 | 11 |

### 2.3 Re-Upload Encoding (Q4) — $L = 4$ Uploads

`AngleEmbedding` is applied once before each of the 4 ansatz layers, multiplying the number of independent $\pm 1/2$ generators to $L \cdot n_\text{qubits}$:

$$
k \in [-L \cdot n, +L \cdot n], \quad \text{upper bound: } 2Ln + 1 = 2 \cdot 4 \cdot 3 + 1 = 25 \text{ components}
$$

**Confirmed prediction:** Q4 consistently shows the widest measured spectrum across all three PDEs (7–8 significant bins vs. 4–7 for angle-encoded configs). The gap between 25 theoretical components and 7–8 observed is expected — random initialisation concentrates Fourier weight in low-frequency components; high-frequency terms are accessible but carry vanishingly small coefficients.

### 2.4 Falsifiable Prediction — Confirmed

Since the **ansatz** only reshapes coefficient magnitudes within $\Omega$ (not the frequency set itself), switching `BasicEntanglerLayers → StronglyEntanglingLayers` at fixed qubit count (Q1 → Q5) should produce near-identical frequency ranges. **Confirmed by data:** Q1 (0.635 Hz, 5 bins) and Q5 (0.477–0.635 Hz, 4–5 bins) are close across all three PDEs, while Q4 reupload consistently dominates.

---

## 3. Barren-Plateau Gradient-Variance Analysis

For a sufficiently expressive, randomly-initialised ansatz approximating a unitary 2-design (McClean et al. [4]):

$$
\text{Var}_{\boldsymbol{\theta}}\!\left[\frac{\partial\langle\hat{M}\rangle}{\partial\theta_k}\right] \sim \mathcal{O}(2^{-n})
$$

### 3.1 Measured Results (McClean-Style Scan)

| Config | Burgers | Heat | NS |
|--------|---------|------|----|
| Q1 (3q) | 0.2586 | 0.2240 | 0.2464 |
| Q2 (4q) | **0.0344** | **0.0369** | **0.0287** |
| Q3 (5q) | 0.0345 | 0.0452 | 0.0321 |
| Q4 (reupload) | 0.0790 | 0.0765 | 0.0820 |
| Q5 (strong ent.) | 0.0836 | 0.0868 | 0.0601 |

**Most reproducible mechanistic finding in the project:** sharp ~7–8× variance drop from 3→4 qubits, plateau 4→5, reproduced across all three PDEs with architecturally identical circuits.

### 3.2 Qualification

The qubit range tested (3–5) is too narrow to confirm exponential collapse vs. other rapidly-decaying trends. The honest claim is: *consistent, sharp variance decline with added qubits, directionally consistent with barren-plateau onset* — **not** confirmed exponential $\mathcal{O}(2^{-n})$ behaviour. At `n_qlayers=4` depth, `BasicEntanglerLayers` circuits of 4–5 qubits have likely not reached the depth/width regime where full 2-design statistics apply.

---

## 4. Weight-Movement Diagnostic

Final weight standard deviation and L2 norm are tracked per-config to distinguish "circuit trained" from "circuit near-frozen." This diagnostic was critical for Q5:

| PDE | Q5 Weight Change | Interpretation |
|-----|-----------------|----------------|
| Burgers | Small | Quantum layer near-frozen |
| Heat | 3.33% | "Weights barely moved" (explicitly flagged in notebook) |
| **NS** | **7.30%** | "Weights moved meaningfully — quantum layer participated actively" |

Same fixed architecture, qualitatively different training regime as a function of PDE loss landscape — a genuine, unresolved-mechanism finding. No classical Tanh layer of fixed width shows this PDE-dependence of active-vs-frozen training.

---

## 5. Design Methodology — Circuit Selection Rules

Synthesised from §7.1 of the Technical Report:

1. **Start with the smallest circuit** (3 qubits, `angle`, `basic_entangler`) — most reliably trainable; only config with confirmed Welch-significant advantage on the primary Burgers' benchmark.
2. **Increase qubit count only against a diagnosed classical weakness** — more qubits widens expressivity but costs gradient-variance headroom (§3.1). C0's pressure-field weakness on NS is the canonical example.
3. **Use the weight-movement check as a pre-registration diagnostic** — if weights stay near-frozen early in training, `strongly_entangling` is not being used; fall back to `basic_entangler`.
4. **Reserve data-reuploading for spectrum-limited cases** — Q4 consistently widens the accessible Fourier spectrum but does not reliably improve accuracy; treat it as a targeted tool, not a general lever.

---

## References

[1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. *Physics-Informed Neural Networks.* J. Computational Physics, 2019.

[2] Shah, N., Lineswala, P., & Chopra, A. *Benchmarking QA-PINN for CFD.* IEEE Quantum Week (QCE), 2024.

[3] Schuld, M., Sweke, R., & Meyer, J. J. *Effect of data encoding on the expressive power of variational quantum ML models.* Physical Review A, 103, 032430 (2021).

[4] McClean, J. R., Boixo, S., Neven, H., & Babbush, R. *Barren plateaus in quantum neural network training landscapes.* Nature Communications, 9, 4812 (2018).
