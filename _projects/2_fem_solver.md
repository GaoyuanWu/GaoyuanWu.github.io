---
layout: page
title: FEM Solver for Structural Dynamics
description: A finite element solver for 2D frame systems under static and dynamic loads.
img: assets/img/solver_sap.png
importance: 2
category: software
---

This was my final project for APC523 (Numerical Algorithms for Scientific Computing). It can be used for the following problems:

1. 2D-frame system under static loads.
2. 2D-frame system under dynamic loads (such as earthquake motions).

A comparison between the solver and the commercial software SAP2000 is shown below, in which the nodal displacement from the solver matches well with SAP2000:

{% include figure.liquid loading="eager" path="assets/img/solver_sap.png" title="Solver vs. SAP2000 comparison" class="img-fluid rounded z-depth-1" %}

For more, please visit the [GitHub repo](https://github.com/GaoyuanWu/NumericalAlgorithms4StructuralDynamics).
