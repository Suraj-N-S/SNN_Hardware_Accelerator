# FPGA-Based Spiking Neural Network Hardware Accelerator

**Bhargav K & Suraj N — IIT Hyderabad**

This project is an FPGA implementation of a Spiking Neural Network (SNN) accelerator, replicating the architecture proposed in:

> **Ali et al., "Energy-Aware FPGA Implementation of Spiking Neural Network with LIF Neurons"**
> arXiv:2411.01628, November 2024
> [https://arxiv.org/abs/2411.01628](https://arxiv.org/abs/2411.01628)

---

## What the paper does (and what we do differently)

The original paper by Ali, Navardi, and Mohsenin (Johns Hopkins University) proposes a hardware-friendly SNN architecture based on the **1st Order Leaky Integrate-and-Fire (LIF) neuron model**, implemented on a **Xilinx Artix-7 FPGA**. Their design targets a **collision avoidance dataset** (real-time vision-based input) and achieves 86% better energy efficiency compared to a Binarized CNN baseline — with no floating-point arithmetic anywhere in the hardware.

**This repo replicates the core SNN architecture but uses the MNIST handwritten digit dataset instead of the collision avoidance dataset.** Everything else — LIF neuron model, fixed-point arithmetic, FPGA deployment on Artix-7 — follows the same approach as the paper.

---

 A. H., Navardi, M., & Mohsenin, T. (2024).
*Energy-Aware FPGA Implementation of Spiking Neural Network with LIF Neurons.*
arXiv:2411.01628. https://arxiv.org/abs/2411.01628
