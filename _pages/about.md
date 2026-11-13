---
layout: about
title: Home
permalink: /
subtitle: Postdoctoral Fellow, Johns Hopkins University

profile:
  align: right
  image: avatar.jpg
  image_circular: false # crops the image to make it circular
  more_info: >
    <p><a href="mailto:gwu37@jh.edu">gwu37@jh.edu</a></p>

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

Hi! My name is Gaoyuan Wu, a Postdoctoral Fellow at Johhns Hopkins University, working with Prof. [Jan Drgona](https://solaris-jhu.github.io/solaris-website/) on differentiable programming and scientific machine learning for optimization and control of systems. I recently obtained my PhD in civil engineering from Princeton University, advised by Prof. [Maria Garlock](https://garlock.princeton.edu). Prior to Princeton, I did my undergraduate studies in civil engineering at Tongji University with a focus in bridge engineering.

I am from Chengdu (成都), a city in southwest China. I love our dialects, our amazing food and, of course, pandas! In my free time, I enjoy a variety of sports — tennis, soccer, badminton, and indoor rock climbing. I love listening to music — I am a proud top 1% listener of Wu Bai and Faye Wong, according to Spotify.

<div style="clear: both;"></div>

<style>
  .pub-cta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.25rem;
    margin: 2rem 0 2.5rem;
    padding: 1.25rem 1.5rem;
    border: 1px solid color-mix(in srgb, var(--global-theme-color) 50%, transparent);
    border-left: 5px solid var(--global-theme-color);
    border-radius: 12px;
    background: color-mix(in srgb, var(--global-theme-color) 7%, transparent);
    box-shadow: 0 0 18px color-mix(in srgb, var(--global-theme-color) 12%, transparent);
  }
  .pub-cta-kicker {
    display: block;
    margin-bottom: 0.3rem;
    color: var(--global-theme-color);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .pub-cta-text {
    margin: 0;
    font-size: 1rem;
    line-height: 1.55;
  }
  a.pub-cta-button {
    flex: none;
    padding: 0.65rem 1.2rem;
    border-radius: 999px;
    background: var(--global-theme-color);
    color: #fff !important;
    font-weight: 700;
    text-decoration: none;
    white-space: nowrap;
    transition: filter 0.2s ease;
  }
  a.pub-cta-button:hover {
    filter: brightness(1.12);
  }
  @media (max-width: 640px) {
    .pub-cta {
      flex-direction: column;
      align-items: flex-start;
    }
  }
</style>

<div class="pub-cta">
  <div>
    <span class="pub-cta-kicker">Research</span>
    <p class="pub-cta-text">
      My research focuses on <strong>enhancing coastal resilience of infrastructure</strong> and
      <strong>differentiable physics &amp; machine learning for optimization</strong>. Please explore my papers here.
    </p>
  </div>
  <a class="pub-cta-button" href="{{ '/research/' | relative_url }}">Explore my research &rarr;</a>
</div>
