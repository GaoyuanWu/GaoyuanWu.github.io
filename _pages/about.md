---
permalink: /
title: "About"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi y'all! I am Gaoyuan (Barry) Wu (吴高远), a Ph.D. student at [Princeton University](https://www.princeton.edu). I am currently working with Prof. [Maria Garlock](https://garlock.princeton.edu). My current research focuses on exploring new structural solutions (such as thin-shell structures) to coastal defense using numerical simulations. I am also interested in leveraging high-performance computing (HPC), accelerated linear algebra (XLA) and machine learning (ML) to boost structural shape optimization.

Before coming to Princeton, I spent 4 years in Shanghai at Tongji University, where I did my undergraduate studies in civil engineering with a focus in bridge engineering.

I am from [Chengdu](https://en.wikipedia.org/wiki/Chengdu) (成都), a city in southwest China. I love our dialects, our amazing food and of course, the PANDAS! 


What's New
=====
- Preprint *"A framework for structural shape optimization based on automatic differentiation, the adjoint method and accelerated linear algebra"* is available online. Link [here](https://arxiv.org/abs/2211.15409). Code for this project is available in its [repo](https://github.com/GaoyuanWu/JaxSSO) with a few examples, check them out. **November, 2022**
- Our new work *"A decoupled SPH-FEM analysis of hydrodynamic wave pressure on hyperbolic-paraboloid thin-shell coastal armor and corresponding structural response"* was accepted by *Engineering Structures* and is available online. Link [here](https://doi.org/10.1016/j.engstruct.2022.114738). **August, 2022**


# Projects
## [JaxSSO](https://github.com/GaoyuanWu/JaxSSO): A framework for structural shape optimization based on automatic differentiation (AD) and the adjoint method, enabled by [JAX](https://github.com/google/jax).

### Features
* Automatic differentiation (AD): an easy and accurate way for gradient evaluation. The implementation of AD avoids deriving derivatives manually or trauncation errors from numerical differentiation.
* Acclerated linear algebra (XLA) and just-in-time compilation: these features in JAX boost the gradient evaluation
* Hardware acceleration: run on GPUs and TPUs for **faster** experience.
* Form finding based on finite element analysis (FEA) and optimization theory

Here is an implementation of JaxSSO to form-find a structure inspired by [Mannheim Multihalle](https://mannheim-multihalle.de/en/architecture/) using simple gradient descent. (First photo credit to Daniel Lukac)
![alt text](https://github.com/GaoyuanWu/JaxSSO/blob/main/data/images/MannheimMultihalle.jpg)
![alt text](https://github.com/GaoyuanWu/JaxSSO/blob/main/data/images/MM_opt.jpg)

## FEM solver of structural dynamics problem for 2D frame system

This is my final project for APC523 (Numerical Algorithms for Scientific Computing). It can be used for the following problems:
1. 2D-frame system under static loads.
1. 2D-frame system under dynamic loads (such as earthquake motions)

A comparison between the solver and commercial software SAP2000 is shown below, in which the nodal displacement from the solver matches well with SAP2000:
<img src="/images/SolverSAP.png" alt="drawing" width="600"/>

For more, please visit the [GitHub repo](https://github.com/GaoyuanWu/NumericalAlgorithms4StructuralDynamics).




