# Mathematical Derivations & Theoretical Analysis

## 1. Differential Equation Formulations

### 1.1 1D Heat Equation (Linear Diffusion)

```math
\frac{\partial u}{\partial t}
=
\alpha\frac{\partial^2u}{\partial x^2},
\qquad
x\in[-1,1],\;
t\in[0,1],
\qquad
\alpha=0.1
```

Initial and boundary conditions:

```math
u(x,0)=\sin(\pi x),
\qquad
u(\pm1,t)=0
```

Exact solution (separation of variables, verified with SymPy):

```math
u(x,t)=\sin(\pi x)e^{-\alpha\pi^2t}
```

---

### 1.2 1D Viscous Burgers' Equation (Nonlinear Convective-Diffusive Transport)

```math
\frac{\partial u}{\partial t}
+
u\frac{\partial u}{\partial x}
=
\nu\frac{\partial^2u}{\partial x^2},
\qquad
\nu=\frac{0.01}{\pi},
\qquad
x\in[-1,1],
\;
t\in[0,1]
```

Initial and Dirichlet boundary conditions:

```math
u(x,0)=-\sin(\pi x),
\qquad
u(\pm1,t)=0
```

Exact solution:

The Cole-Hopf transformation reduces Burgers' equation to the heat equation. The reference solution is computed using Gauss-Hermite quadrature and verified symbolically with SymPy.

---

### 1.3 2D Kovasznay Flow (Steady Incompressible Navier-Stokes)

Momentum equations:

```math
u\,u_x+v\,u_y
=
-p_x+\nu(u_{xx}+u_{yy})
```

```math
u\,v_x+v\,v_y
=
-p_y+\nu(v_{xx}+v_{yy})
```

Continuity equation:

```math
u_x+v_y=0
```

where

```math
\nu=\frac1{Re},
\qquad
Re=40
```

The Kovasznay parameter is

```math
\lambda
=
\frac{Re}{2}
-
\sqrt{\frac{Re^2}{4}+4\pi^2}
\approx-0.9637
```

Exact analytical solution (verified with SymPy):

```math
u(x,y)=1-e^{\lambda x}\cos(2\pi y)
```

```math
v(x,y)
=
\frac{\lambda}{2\pi}
e^{\lambda x}
\sin(2\pi y)
```

```math
p(x,y)
=
\frac12
\left(
1-e^{2\lambda x}
\right)
```

---

# 2. Fourier-Frequency Analysis

Following Schuld, Sweke and Meyer [3], the expectation value of a variational quantum circuit with angle encoding admits a finite Fourier expansion

```math
f(\mathbf{x},\boldsymbol{\theta})
=
\langle\hat M\rangle
=
\sum_{\omega\in\Omega}
c_\omega(\boldsymbol{\theta})
e^{i\omega x}
```

The accessible frequency set depends only on the encoding Hamiltonian. The variational ansatz changes the Fourier coefficients
\(c_\omega(\theta)\) but does not introduce frequencies outside \(\Omega\).

---

## 2.1 Encoding Used in This Project

Every QAPINN configuration employs

```text
AngleEmbedding(inputs, rotation="Y")
```

Each qubit applies an

```math
R_Y(x_i)
```

rotation.

The generator

```math
Y/2
```

has eigenvalues

```math
\pm\frac12
```

giving the accessible single-qubit frequencies

```math
\omega_i\in\{-1,0,+1\}
```

obtained from the pairwise eigenvalue differences.

---

## 2.2 Angle Encoding (Q1, Q2, Q3, Q5)

The accessible frequency set is

```math
\Omega=
\left\{
k\in[-n,+n]:
k=\sum_{i=1}^{n}\omega_i,
\;
\omega_i\in\{-1,0,+1\}
\right\}
```

Hence,

```math
|\Omega|\le2n+1.
```

| Configuration | Qubits | Maximum Frequency Components |
|--------------|--------|------------------------------|
| Q1 | 3 | 7 |
| Q2 | 4 | 9 |
| Q3 | 5 | 11 |
| Q5 | 3 | 7 |

---

## 2.3 Data Re-upload Encoding (Q4)

For Q4, AngleEmbedding is applied before each of the four variational layers.

The accessible spectrum becomes

```math
\Omega=
\left\{
k\in[-Ln,+Ln]:
k=\sum_{i=1}^{Ln}\omega_i,
\;
\omega_i\in\{-1,0,+1\}
\right\},
\qquad
L=4.
```

Therefore,

```math
|\Omega|
\le
2Ln+1
=
2\times4\times3+1
=
25.
```

Experimentally, Q4 consistently exhibits the widest measured spectrum across all three PDEs (7-8 significant frequency bins). The difference between the theoretical maximum (25) and the observed spectrum is expected because randomly initialized circuits concentrate most Fourier weight in low-frequency coefficients while higher-order coefficients remain negligible.

---

## 2.4 Experimental Verification

Since the variational ansatz only modifies the Fourier coefficients
\(c_\omega(\theta)\) while leaving the accessible frequency set unchanged, replacing

```text
BasicEntanglerLayers
```

with

```text
StronglyEntanglingLayers
```

at fixed qubit count should produce similar frequency ranges.

This prediction is confirmed experimentally.

| Comparison | Measured Spectrum |
|------------|-------------------|
| Q1 | 0.635 Hz, 5 bins |
| Q5 | 0.477-0.635 Hz, 4-5 bins |
| Q4 | Widest spectrum across all PDEs |

---

# 3. Barren Plateau Gradient Variance

Following McClean et al. [4], sufficiently expressive random quantum circuits satisfy

```math
\operatorname{Var}_{\boldsymbol{\theta}}
\left[
\frac{\partial\langle\hat M\rangle}
{\partial\theta_k}
\right]
=
\mathcal O(2^{-n})
```

---

## 3.1 Measured Gradient Variance

| Configuration | Burgers | Heat | Navier-Stokes |
|--------------|---------|------|---------------|
| Q1 (3 qubits) | 0.2586 | 0.2240 | 0.2464 |
| Q2 (4 qubits) | **0.0344** | **0.0369** | **0.0287** |
| Q3 (5 qubits) | 0.0345 | 0.0452 | 0.0321 |
| Q4 (Re-upload) | 0.0790 | 0.0765 | 0.0820 |
| Q5 (Strongly Entangling) | 0.0836 | 0.0868 | 0.0601 |

The most reproducible observation is the sharp 7-8× reduction in gradient variance when increasing from three to four qubits, followed by a plateau from four to five qubits.

---

## 3.2 Qualification

The investigated range (3-5 qubits) is insufficient to demonstrate exponential decay conclusively.

The appropriate conclusion is:

> A consistent decrease in gradient variance is observed with increasing qubit count, consistent with the onset of barren plateau behaviour, although exponential scaling cannot be established over the limited qubit range investigated.

---

# 4. Weight Movement Diagnostic

The final parameter standard deviation and L2 norm are tracked to distinguish active optimization from nearly frozen quantum layers.

| PDE | Q5 Weight Change | Interpretation |
|-----|-----------------|----------------|
| Burgers | Small | Near-frozen quantum layer |
| Heat | 3.33% | Minimal parameter movement |
| Navier-Stokes | 7.30% | Active quantum participation |

The same quantum architecture therefore exhibits markedly different optimization behaviour depending on the governing PDE.

---

# 5. Circuit Selection Methodology

1. Begin with the smallest circuit (3 qubits, angle encoding, BasicEntanglerLayers).
2. Increase qubit count only when the classical baseline exhibits a measurable limitation.
3. Monitor parameter movement during training. Near-frozen parameters indicate ineffective quantum optimization.
4. Use data re-uploading only when additional Fourier expressivity is required.

---

# References

[1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. *Physics-Informed Neural Networks*. Journal of Computational Physics, 2019.

[2] Shah, N., Lineswala, P., & Chopra, A. *Benchmarking QA-PINN for Computational Fluid Dynamics*. IEEE Quantum Week (QCE), 2024.

[3] Schuld, M., Sweke, R., & Meyer, J. J. *The Effect of Data Encoding on the Expressive Power of Variational Quantum Machine Learning Models*. Physical Review A, 103, 032430 (2021).

[4] McClean, J. R., Boixo, S., Neven, H., & Babbush, R. *Barren Plateaus in Quantum Neural Network Training Landscapes*. Nature Communications, 9, 4812 (2018).
