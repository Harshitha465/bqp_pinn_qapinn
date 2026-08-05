# Quantum-Assisted Physics-Informed Neural Networks for Computational Fluid Dynamics

**WISER Summer Program 2026 – BQP Challenge**

## Challenge Statement

The challenge investigates how the introduction of a **quantum layer** influences the learning process of **Physics-Informed Neural Networks (PINNs)**. Rather than demonstrating that Quantum-Assisted PINNs (QAPINNs) simply outperform classical PINNs, the focus is on understanding:

- When the quantum layer changes learning behavior.
- Why these changes occur.
- How quantum circuits should be designed for different classes of partial differential equations (PDEs).
- What benefits and trade-offs arise from incorporating a quantum layer.

## Project Overview

This repository presents our implementation and analysis of **Quantum-Assisted Physics-Informed Neural Networks (QAPINNs)** developed for the **WISER Summer Program 2026 – BQP Challenge**. The project investigates how replacing part of a classical Physics-Informed Neural Network (PINN) with a **Variational Quantum Circuit (VQC)** influences the learning process when solving partial differential equations (PDEs).

Rather than focusing solely on whether QAPINNs outperform classical PINNs, this work aims to understand **when**, **why**, and **how** the introduction of a quantum layer affects learning behavior. To achieve this, we performed a systematic comparison between classical PINNs and multiple QAPINN configurations across three benchmark PDEs: **Burgers' Equation, Heat Equation, and Navier–Stokes (Kovasznay Flow)**.

In addition to evaluating predictive performance, the project investigates the impact of different quantum circuit designs—including qubit count, data encoding strategies, entanglement structures, and circuit architectures—using explainability techniques such as **Fourier spectrum analysis, gradient variance analysis, and activation map visualization**. The repository includes the complete implementation, experimental results, technical report, presentation slides, and documentation required to reproduce the study.

## Objectives

The objectives of this project are to:

- Investigate how the introduction of a quantum layer influences the learning process of Physics-Informed Neural Networks (PINNs).
- Compare the learning behavior and predictive performance of Classical PINNs and Quantum-Assisted PINNs (QAPINNs) across benchmark partial differential equations.
- Evaluate the influence of quantum circuit design choices, including qubit count, data encoding strategies, and variational ansatz architectures.
- Analyze learning dynamics using Explainable AI (XAI) techniques, including Fourier spectrum analysis, gradient variance analysis, and activation map visualization.
- Assess the effectiveness of QAPINNs using quantitative performance metrics such as Relative L2 Error, PDE residual error, training time, and model complexity.
- Develop insights and design guidelines for constructing problem-specific quantum-enhanced architectures for solving partial differential equations.

## Repository Structure

```text
bqp_pinn_qapinn/
│
├── README.md                          # Project overview and documentation
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git ignore rules
│
├── models/
│   ├── pinn.py                        # Classical PINN implementation
│   ├── qapinn.py                      # Quantum-Assisted PINN implementation
│   └── quantum_layers.py              # Variational Quantum Circuit (VQC) layers
│
├── notebooks/
│   ├── Burgers_Equation_JAX.ipynb     # Burgers' Equation experiments
│   ├── Heat_Equation_JAX.ipynb        # Heat Equation experiments
│   └── NavierStokes_Kovasznay.ipynb   # Kovasznay Flow (Navier–Stokes) experiments
│
├── results/
│   ├── Burgers/
│   │   ├── master_performance_table.png
│   │   ├── prediction_vs_exact.png
│   │   ├── fourier_spectrum.png
│   │   ├── activation_maps.png
│   │   ├── training_loss.png
│   │   └── metrics.csv
│   │
│   ├── Heat/
│   │   ├── master_performance_table.png
│   │   ├── prediction_vs_exact.png
│   │   ├── fourier_spectrum.png
│   │   ├── activation_maps.png
│   │   ├── training_loss.png
│   │   └── metrics.csv
│   │
│   └── Navier_Stokes/
│       ├── master_performance_table.png
│       ├── prediction_vs_exact.png
│       ├── fourier_spectrum.png
│       ├── activation_maps.png
│       ├── training_loss.png
│       └── metrics.csv
│
├── docs/
│   ├── methodology.md
│   ├── mathematical_analysis.md
│   └── reproducibility.md
│
├── REPORT.pdf                         # Technical report
├── PRESENTATION.pptx                  # Project presentation
└── LICENSE                            # License
```

## Experimental Setup

The experiments were designed to evaluate the impact of introducing a **Variational Quantum Circuit (VQC)** as a replacement for the first hidden layer of a classical Physics-Informed Neural Network (PINN). Classical and quantum-assisted models were trained and evaluated under comparable settings across multiple benchmark partial differential equations.

### Benchmark Problems

The following benchmark PDEs were considered:

- **Burgers' Equation** – A nonlinear PDE used to study shock formation and nonlinear wave propagation.
- **Heat Equation** – A linear diffusion equation used to evaluate model behavior on smooth solutions.
- **Navier–Stokes (Kovasznay Flow)** – A two-dimensional steady-state fluid dynamics benchmark used to assess model performance on complex nonlinear systems.

### Model Configurations

Six model configurations were evaluated throughout the study.

| Configuration | Description |
|---------------|-------------|
| **C0** | Classical Physics-Informed Neural Network (PINN) |
| **Q1** | 3-Qubit, Angle Encoding, Basic Entangler |
| **Q2** | 4-Qubit, Angle Encoding, Basic Entangler |
| **Q3** | 5-Qubit, Angle Encoding, Basic Entangler |
| **Q4** | 3-Qubit, Data Re-uploading, Basic Entangler |
| **Q5** | 3-Qubit, Angle Encoding, Strongly Entangling Ansatz |

### Technologies Used

- **Programming Language:** Python
- **Machine Learning Framework:** JAX
- **Quantum Machine Learning Framework:** PennyLane
- **Optimization Library:** Optax
- **Scientific Computing:** NumPy, SciPy
- **Visualization:** Matplotlib

### Execution Environment

- Google Colab
- PennyLane `default.qubit` quantum simulator
- JAX JIT compilation for accelerated execution

### Evaluation Metrics

Model performance was evaluated using the following metrics:

- Relative L2 Error
- PDE Residual Error
- Training Time
- Number of Trainable Parameters
- Fourier Spectrum Analysis
- Gradient Variance Analysis
- Activation Map Visualization

  ## Results Summary

The experimental study demonstrates that the effectiveness of Quantum-Assisted Physics-Informed Neural Networks (QAPINNs) is highly dependent on both the underlying partial differential equation and the chosen quantum circuit architecture. Rather than consistently outperforming classical PINNs, the introduction of a quantum layer alters the learning dynamics in ways that vary across different problem domains.

The key findings of this study include:

- The 5-qubit angle-encoded QAPINN (Q3) achieved the strongest statistically significant improvement over the classical PINN on the Navier–Stokes (Kovasznay Flow) benchmark.
- Data re-uploading consistently increased the accessible Fourier spectrum, indicating greater representational capacity, but did not consistently translate into improved predictive performance.
- Increasing the number of qubits reduced gradient variance, highlighting a trade-off between model expressivity and trainability.
- The strongly entangling ansatz exhibited PDE-dependent training behavior, demonstrating that the effectiveness of a quantum circuit depends on the characteristics of the underlying differential equation.
- Overall, the results indicate that quantum layers provide problem-specific advantages rather than serving as a universal replacement for classical PINNs.

  ## Installation

Clone the repository:

```bash
git clone https://github.com/Harshitha465/bqp_pinn_qapinn.git
cd bqp_pinn_qapinn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
