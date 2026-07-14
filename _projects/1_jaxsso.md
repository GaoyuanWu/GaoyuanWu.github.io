---
layout: page
title: JaxSSO
description: A differentiable finite element analysis (FEA) solver for structural optimization, enabled by JAX.
img: assets/img/mannheim_multihalle.jpg
importance: 1
category: software
related_publications: true
---

[JaxSSO](https://github.com/GaoyuanWu/JaxSSO) is a differentiable finite element analysis (FEA) solver for structural optimization, enabled by [JAX](https://github.com/google/jax).

#### Features

- Automatic differentiation (AD): an easy and accurate way for gradient evaluation. The implementation of AD avoids deriving derivatives manually or truncation errors from numerical differentiation.
- Accelerated linear algebra (XLA) and just-in-time compilation: these features in JAX boost the gradient evaluation.
- Hardware acceleration: run on GPUs and TPUs for a faster experience.
- Support for beam-column elements and MITC-4 quadrilateral shell elements.
- Shape optimization, size optimization, and topology optimization.
- Seamless integration with machine learning (ML) libraries.

See {% cite wu2023framework %} and {% cite wu2024jaxsso %} for the underlying methodology and software.

Here is an implementation of JaxSSO to form-find a structure inspired by [Mannheim Multihalle](https://mannheim-multihalle.de/en/architecture/) using simple gradient descent. (First photo credit to Daniel Lukac.)

<div class="row justify-content-sm-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/mannheim_multihalle.jpg" title="Mannheim Multihalle" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="row justify-content-sm-center">
    <div class="col-sm-11 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/mm_opt.jpg" title="JaxSSO form-finding result" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
