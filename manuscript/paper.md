---
title: 'pyRTX: a Python package high precision computation of non gravitational forces on deep space probes'
tags:
  - Python
  - astodynamics
  - solar pressure
  - atmospheric drag
  - ray tracing
authors:
  - name: Gael Cascioli
    corresponding: true # (This is how to denote the corresponding author)
    orcid: 0000-0001-9070-7947
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Ariele Zurria
    orcid: 0009-0003-6155-7160
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 3
  - name: Erwan Mazarico
    orcid: 0000-0003-3456-427X
    affiliation: 2
affiliations:
 - name: University of Maryland Baltimore County, United States
   index: 1
   ror: 00hx57361
 - name: NASA Goddard Space Flight Center, United States
   index: 2
 - name: Sapienza University, Italy
   index: 3
date: 3 December 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: The Planetary Science Journal <- The name of the AAS journal.
---

# Summary

With the constant improvement of radiometric tracking systems, inaccuracies
in the non-gravitational force modeling have become one of the limiting factors
to precise orbit determination, and the scientific products that it enables. 
The main factor impacting the limited accuracy of non-gravitational force models
is the complex 3D shape of the spacecraft. While fast, reliable,  analytical models
are available for simple shapes (spheres, cubes, etc), no such model is generally
available for a complex shape. This software package aims to address this limitation by legeraging ray-tracing to compute the complex interaction between the forcing environment (radiation, atmosphere) and the three dimensional shape of the spacecraft. 





# Statement of need

Several scientific investigations require high-precision reconstruction of 
spacecraft trajectories. Among these, one of the most demanding is the determination
of the gravity field of Solar System bodies (planets, moons). This task is accomplished
by solving the so-called orbit determination (OD) problem [@tapley_statistical_2004,milani_theory_2009]. The solution of the OD in the adjustment of a dynamical model 
(a set of differential equations) describing the spacecraft motion. Systematic errors
in the dynamical model will almost inevitably lead to systematic errors in the solution. 
In the recent years significant improvements in radiometric tracking system, have led
to more and more precise measurements of the spacecraft position and velocity (the input to the OD), thus requiring increasingly more accurate dynamical models [@cappuccio_report_2020,asmar_spacecraft_2005,mazarico_europa_2023]. 
One of the major limitations of current dynamical modelling of deep-space probes consists in the complex interaction between the spacecraft shape and the atmosphere, and with radiative forces (solar radiation pressure, albedo, thermal infrared radiation). We developed the pyRTX software package to address this limitation.
Leveraging the ray-tracing technique, originally developed in computer graphics, instead of relying on simplified macro-models (flat plates, cylinders, etc) pyRTX is able to use the actual 3D shape of the spacecraft (provided as, for example, an .obj file). 



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
