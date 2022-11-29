---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

# Journal
[1] **G. Wu**, M. Garlock, and S. Wang, *A decoupled SPH-FEM analysis of hydrodynamic wave pressure on hyperbolic-paraboloid thin-shell coastal armor and corresponding structural response*, Engineering Structures, vol. 268, p. 114738, Oct. 2022, [doi: 10.1016/j.engstruct.2022.114738](https://doi.org/10.1016/j.engstruct.2022.114738)

<img src="/images/Hypar.jpg" alt="drawing" width="600"/>
* This work studies the interaction between waves and Kinetic Umbrellas, a newly proposed structural solution for coastal hazard mitigation. A decoupled numerical scheme consisting of Smoothed Particle Hydrodynamics (SPH) and Finite Element Modeling (FEM) is illustrated and implemented to study the wave pressure and corresponding structural response. The results show that hypar thin shells are structurally feasible under surge and wave loadings, which can serve as a sustainable alternative to traditional coastal barriers that are uni-functional and structurally inefficient. The presented framework for studying wave-shell interaction can be generalized for various structural geometries.

# Preprint
[1] **G. Wu**, *A framework for structural shape optimization based on automatic differentiation, the adjoint method and accelerated linear algebra*, arXiv Preprint, Nov. 2022, [arXiv:2211.15409](https://doi.org/10.48550/arXiv.2211.15409)

<img src="/images/MannheimMultihalle.jpg" alt="drawing" width="800"/>
<img src="/images/MM_opt.jpg" alt="drawing" width="800"/>

* Shape optimization is of great significance in structural engineering, as an efficient geometry leads to better performance of structures. However, the application of gradient-based shape optimization for structural and architectural design is limited, which is partly due to the difficulty and the complexity in gradient evaluation. In this work, an efficient framework based on automatic differentiation (AD), the adjoint method and accelerated linear algebra (XLA) is proposed to promote the implementation of gradient-based shape optimization. The framework is realized by the implementation of the high-performance computing (HPC) library JAX. We leverage AD for gradient evaluation in the sensitivity analysis stage. Compared to numerical differentiation, AD is more accurate; compared to analytical and symbolic differentiation, AD is more efficient and easier to apply. In addition, the adjoint method is used to reduce the complexity of computation of the sensitivity. The XLA feature is exploited by an efficient programming architecture that we proposed, which can boost gradient evaluation. The proposed framework also supports hardware acceleration such as GPUs. The framework is applied to the form finding of arches and different free-form gridshells: gridshell inspired by Mannheim Multihalle, four-point supported gridshell, and canopy-like structures. Two geometric descriptive methods are used: non-parametric and parametric description via Bézier surface. Non-constrained and constrained shape optimization problems are considered, where the former is solved by gradient descent and the latter is solved by sequential quadratic programming (SQP). Through these examples, the proposed framework is shown to be able to provide structural engineers with a more efficient tool for shape optimization, enabling better design for the built environment.