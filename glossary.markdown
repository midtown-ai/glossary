---
title: Glossary

# Location of index.html
# <!> If not provided, uses <filename>.html
permalink: /
# permalink: /glossary/

# === THEMES
# = MINIMA
layout: page
# layout: home

# = MINIMAL-MISTAKE
# layout: archive-taxonomy
# layout: archive
# layout: categories
# layout: category
# layout: collection
# layout: compress
# layout: default
# layout: home
# layout: posts
# layout: search
## layout: single
# layout: splash
# layout: tag
# layout: tags

pdf:
  file: "files/pdf/musiclm_model_paper.pdf"
---

# MY PDFS

* unordered list
{:toc}

## VIDEO 0

{% youtube "https://www.youtube.com/watch?v=bYAZLSysoqI" %}

See also [PDF2](#PDF-2)
See also [PDF2](#pdf-2)
See also [PDF2](#jump-here)

## PDF 0

This is the base Jekyll theme. You can find out more info about customizing your Jekyll theme, as well as basic Jekyll usage documentation at [jekyllrb.com](https://jekyllrb.com/)

You can find the source code for Minima at GitHub:
[jekyll][jekyll-organization] /
[minima](https://github.com/jekyll/minima)

You can find the source code for Jekyll at GitHub:
[jekyll][jekyll-organization] /
[jekyll](https://github.com/jekyll/jekyll)


# requires kramdown in _config

{:auto_ids}

term
: defaintion

term2
: defaintion2

## PDF 1
{% pdf {{ page.pdf.file }} %}
## PDF 2
{% pdf {{ page.pdf.file }} %}
## PDF 3
{% pdf {{ page.pdf.file }} %}

## jump here


[jekyll-organization]: https://github.com/jekyll
