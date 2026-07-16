---
layout: about
title: Home
permalink: /
subtitle: Postdoctoral Research Associate, Princeton University

profile:
  align: right
  image: avatar.jpg
  image_circular: false # crops the image to make it circular
  more_info: >
    <p><a href="mailto:gaoyuanw@princeton.edu">gaoyuanw@princeton.edu</a></p>

selected_papers: false # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: true # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 2 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

<style>
  .profile {
    width: 220px !important;
  }
</style>

Hi! My name is Gaoyuan Wu, a Postdoctoral Research Associate at Princeton University. I recently obtained my PhD in civil engineering from Princeton University, advised by Prof. [Maria Garlock](https://garlock.princeton.edu). Prior to Princeton, I did my undergraduate studies in civil engineering at Tongji University with a focus in bridge engineering.

I am from Chengdu (成都), a city in southwest China. I love our dialects, our amazing food and, of course, pandas! In my free time, I enjoy a variety of sports — tennis, soccer, badminton, and indoor rock climbing. I love listening to music — I am a proud top 1% listener of Wu Bai and Faye Wong, according to Spotify.

<div style="clear: both;"></div>

## Research

For a full list of publications, please see my [publications page](/publications/) or my [Google Scholar profile](https://scholar.google.com/citations?user=2aTANLwAAAAJ). A few selected projects are highlighted below.

### Enhancing coastal resilience of infrastructure

My research integrates structural engineering, coastal engineering, and optimization technique (e.g., Bayesian Optimization) to make infrastructure more resilient against extreme wave loads induced by climate-influenced hazards.

<div class="row justify-content-sm-center">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/Side_Hypar.gif" title="Deployable thin-shell coastal armor under wave loading" class="img-fluid rounded z-depth-1" avoid_scaling=true caption="Deployable thin-shell coastal armor under wave loading (<a href='/publications/#wu2022decoupled'>Wu et al., 2022</a>)." %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/BO_flow.png" title="Multi-objective Bayesian Optimization framework" class="img-fluid rounded z-depth-1" caption="Multi-objective Bayesian Optimization framework for wave-load shape optimization of structures (<a href='/publications/#wu2026bayes'>Wu et al., 2026</a>)." %}
  </div>
</div>

### Differentiable finite element analysis for structural optimization and integration with machine learning

[JaxSSO](https://github.com/GaoyuanWu/JaxSSO) is a *differentiable* finite element analysis (FEA) solver built on JAX, enabling gradient-based structural optimization that reduces embodied carbon while achieving structural elegance ([Wu, 2023](/publications/#wu2023framework); [Wu, 2024](/publications/#wu2024jaxsso)). It also enables training of neural networks (NN) with physics enforced in the loop.

<div class="row justify-content-sm-center">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/jaxsso.png" title="JaxSSO" class="img-fluid rounded z-depth-1" caption="Training of NN with physics (enforced via FEA) in the loop" %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/mm_opt.jpg" title="JaxSSO form-finding result" class="img-fluid rounded z-depth-1" caption="JaxSSO form-finding a gridshell inspired by the Mannheim Multihalle." %}
  </div>
</div>

For more on this work, see the [JaxSSO project page](/projects/1_jaxsso/) and [codes page](/codes/).
