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
