---
layout: page
permalink: /publications/
title: Publications
description: 
nav: true
nav_order: 2
redirect_from:
  - /research/
---

<!-- _pages/publications.md -->

<!-- Darken the year headings; al_folio_core defaults them to the very light
     --global-divider-color, which keeps the divider rule but washes out the text. -->
<style>
  .publications h2.bibliography {
    color: var(--global-text-color-light);
  }
</style>

<!-- Bibsearch Feature -->

{% include bib_search.liquid %}

<div class="publications">

{% bibliography %}

</div>
