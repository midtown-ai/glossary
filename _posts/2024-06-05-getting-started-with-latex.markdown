---
layout: post
# layout: single
title:  "Getting Started with Latex"
date:   2024-06-05 12:51:28 -0800
categories: jekyll update
---

{% include links/all.md %}

* toc
{:toc}


## Links

 * Latex classes - [https://www.ctan.org/topic/class](https://www.ctan.org/topic/class)
 * packages
   * natbib - bibliography - [https://www.ctan.org/pkg/natbib](https://www.ctan.org/pkg/natbib)
   * tables - [https://www.ctan.org/pkg/booktabs](https://www.ctan.org/pkg/booktabs)
 * Tutorials
   * overleaf - [https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes)
 * reformating of latex
   * to pdf with pdflatex - [https://www.wavewalkerdsp.com/2022/01/05/install-and-run-latex-on-macos/](https://www.wavewalkerdsp.com/2022/01/05/install-and-run-latex-on-macos/)
   * to html with jekyll - [https://github.com/mhartl/jekyll-latex](https://github.com/mhartl/jekyll-latex)

## Documentation

 {% pdf "https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh.pdf" %}

## Command


 :warning: You can set the auto-recompile on

```
Cmd + Enter                          - recompile
```

## References

```
                                     % BEGINNING OF PREAMPLE

% \documentclass{article}            % Class --> control overall appearance of the document
                                     % = article, book, report, CV/resume

\documentclass[12pt, letterpaper]{article}  % Class with parameters (font and paper size)
                                     % default font size = 10pt
                                     % default paper size = a4paper

\usepackage{graphicx}                % Package required for inserting images

\title{tests}                        % Metadata used to generate title
\author{Emmanuel Mayssat}
\date{June 2024}

                                     % END OF PREAMPLE

\begin{document}                     % _begin_document = begin the body of the document

\maketitle                           % Generate Title using provided metadata

\section{Heading; 1st level}         % Header-1

\subsection{Heading: 2nd level}      % Header-2

\subsubsection{Heading: 3rd level}   % Header-3

Non-indented Normal text             % Normal text
This is a simple example.
This is on the same line as above.

Indented text line 1

Indented text line 2

writen in \LaTeX{}                   % Macro => Insert the Latex icons
or \LaTeXe{}

\textbf{bold text}
\emph{emphasis with italics}

reference \ref{anchor_1}             % Reference => jump to a labeled achor in the same document
\label{anchor_1}


Inline raw \verb+\paragraph+ command % inline raw text

\begin{verbatim}                     % raw text
   \citet{hasselmo} investigated\dots
\end{verbatim}



% \usepackage{url}                   % Package: simple URL typesetting
\begin{center}                       % centered section

  \url{http://mirrors.ctan.org/macros/latex/contrib/natbib/natnotes.pdf}  % clickable URL
\end{center}


\begin{figure}                       % declare a floating figure
  \centering
  \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}} % in a box
  \caption{Sample figure caption.}
\end{figure}


% \usepackage{booktabs}              % professional-quality tables
Used to typeset Table~\ref{sample-table}.

\begin{table}
  \caption{Sample table title}       % Table caption
  \label{sample-table}
  \centering
  \begin{tabular}{lll}
    \toprule
    \multicolumn{2}{c}{Part}                   \\
    \cmidrule(r){1-2}
    Name     & Description     & Size ($\mu$m) \\
    \midrule
    Dendrite & Input terminal  & $\sim$100     \\
    Axon     & Output terminal & $\sim$10      \\
    Soma     & Cell body       & up to $10^6$  \\
    \bottomrule
  \end{tabular}
\end{table}

\end{document}                       % _end_document
```
