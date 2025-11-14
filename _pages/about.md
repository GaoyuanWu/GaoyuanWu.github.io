---
permalink: /
title: "About"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi all! I am Gaoyuan Wu (吴高远), a Ph.D. student at Princeton University. I am currently working with Prof. [Maria Garlock](https://garlock.princeton.edu). My current research focuses on exploring new structural solutions (such as thin-shell structures) to coastal defense using numerical simulations. I am also interested in leveraging the high-performance computing (HPC) Python library [JAX](https://github.com/google/jax) to boost structural shape optimization. Outside of the research world, I served as an [Residential Graduate Student](https://forbescollege.princeton.edu/people/residential-graduate-students-rgs) (RGS) at Forbes College for two years, from 2023 to 2025.

Prior to Princeton, I did my undergraduate studies in civil engineering at Tongji University with a focus in bridge engineering.


What's New
=====
- Our new work *"An SPH study of cross-sectional shape effects on coastal structures subject to regular wave forces* and is available online. DOI: [10.1016/j.oceaneng.2025.122929](https://doi.org/10.1016/j.oceaneng.2025.122929). **September, 2025**
- Preprint *"JAX-SSO: Differentiable Finite Element Analysis Solver for Structural Optimization and Seamless Integration with Neural Networks"* is available online. arXiv link: [arXiv:2407.20026](
https://doi.org/10.48550/arXiv.2407.20026).  **July 2024**
- Our new work *"Investigating the Effects of Box Girder Bridge Geometry on Solitary Wave Force Using SPH Modeling"* was accepted by *Coastal Engineering* and is available online. DOI: [10.1007/s00158-023-03601-0](https://doi.org/10.1016/j.coastaleng.2023.104430). PDF is available [here](https://authors.elsevier.com/a/1i8-h1M2DVKRZ4). **November, 2023**
- Our new work *"A framework for structural shape optimization based on automatic differentiation, the adjoint method and accelerated linear algebra"* was accepted by *Structural and Multidisciplinary Optimization* and is available online. DOI: [10.1007/s00158-023-03601-0](http://dx.doi.org/10.1007/s00158-023-03601-0). PDF is available [here](https://rdcu.be/deZ2W). Code for this project is available in its [repo](https://github.com/GaoyuanWu/JaxSSO) with a few examples, check them out. **June, 2023**



# Projects
## [JaxSSO](https://github.com/GaoyuanWu/JaxSSO): A differentiable finite element analysis (FEA) solver for structural optimization, enabled by [JAX](https://github.com/google/jax).

### Features
* Automatic differentiation (AD): an easy and accurate way for gradient evaluation. The implementation of AD avoids deriving derivatives manually or trauncation errors from numerical differentiation.
* Acclerated linear algebra (XLA) and just-in-time compilation: these features in JAX boost the gradient evaluation
* Hardware acceleration: run on GPUs and TPUs for **faster** experience
* Support beam-column elements and MITC-4 quadrilateral shell elements
* Shape optimization, size optimization and topology optimization
* Seamless integration with machine learning (ML) libraries

Here is an implementation of JaxSSO to form-find a structure inspired by [Mannheim Multihalle](https://mannheim-multihalle.de/en/architecture/) using simple gradient descent. (First photo credit to Daniel Lukac)

<img src="/images/MannheimMultihalle.jpg" alt="drawing" width="800"/>
<img src="/images/MM_opt.jpg" alt="drawing" width="800"/>

## FEM solver of structural dynamics problem for 2D frame system

This is my final project for APC523 (Numerical Algorithms for Scientific Computing). It can be used for the following problems:
1. 2D-frame system under static loads.
1. 2D-frame system under dynamic loads (such as earthquake motions)

A comparison between the solver and commercial software SAP2000 is shown below, in which the nodal displacement from the solver matches well with SAP2000:
<img src="/images/SolverSAP.png" alt="drawing" width="600"/>

For more, please visit the [GitHub repo](https://github.com/GaoyuanWu/NumericalAlgorithms4StructuralDynamics).




