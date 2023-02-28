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


{% raw %}

	// your code here

{% endraw %}

{% highlight html linenos %}

	// your code here

{% endhighlight %}

# MY PDFS

* unordered list
{:toc}


{% for p in site.pages %}
  {{ p.path }}
{% endfor %}

Link to [Section 1]
or [Section 2]
or [SEction 2]

## VIDEO 0

{% youtube "https://www.youtube.com/watch?v=bYAZLSysoqI" %}

See also [PDF2](#PDF-2)
See also [PDF2](#pdf-2)
See also [Jump](#jump-here)
See also [About](/about/#here)
see also [External]
see also [Artificial Intelligence]
see also [Term]
see also [here](#term)

## PDF 0

This is the base Jekyll theme. You can find out more info about customizing your Jekyll theme, as well as basic Jekyll usage documentation at [jekyllrb.com](https://jekyllrb.com/)

You can find the source code for Minima at GitHub:
[jekyll][jekyll-organization] /
[minima](https://github.com/jekyll/minima)

You can find the source code for Jekyll at GitHub:
[jekyll][jekyll-organization] /
[jekyll](https://github.com/jekyll/jekyll)


# requires kramdown in _config

{:auto_ids}  # Add id to DL

term
: defaintion

term2
: defaintion2

## PDF 1
{% pdf {{ page.pdf.file }} %}
## PDF 2
{% pdf {{ page.pdf.file }} %}
## PDF 3
{% pdf "files/pdf/musiclm_model_paper.pdf" %}

## jump here
## External

{% include links.md %}

[jekyll-organization]: https://github.com/jekyll
