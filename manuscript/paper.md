---
title: 'pyRTX: a Python package for high precision computation of non gravitational forces on deep space probes'
tags:
  - Python
  - astrodynamics
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
aas-journal: The Planetary Science Journal
---

# Summary

With the constant improvement of radiometric tracking systems, inaccuracies
in the non-gravitational force modeling have become one of the limiting factors
to deep space precise orbit determination, and the scientific products that it enables. 
The main factor impacting the limited accuracy of non-gravitational force models
is the complex 3D shape of the spacecraft. While fast, reliable,  analytical models
are available for simple shapes (spheres, flat plates, etc), no such model is generally
available for a complex shape. This software package aims to address this limitation by leveraging ray-tracing to compute the complex interaction between the forcing environment (radiation, atmosphere) and the three dimensional shape of the spacecraft. 
This software is specifically geared towards planetary OD where inaccuracies 
of the model often cannot be overcome with continuous tracking as it is routinely done for Earth-orbiting spacecraft.




# Statement of need

Several scientific investigations require high-precision reconstruction of 
spacecraft trajectories. Among these, one of the most demanding is the determination
of the gravity field of Solar System bodies (planets, moons). This task is accomplished
by solving the so-called orbit determination (OD) problem [@tapley_statistical_2004;@milani_theory_2009]. The solution of the OD in the adjustment of a dynamical model 
(a set of differential equations) describing the spacecraft motion. Systematic errors
in the dynamical model will almost inevitably lead to systematic errors in the solution. 
In the recent years significant improvements in radiometric tracking system, have led
to more and more precise measurements of the spacecraft position and velocity (the input to the OD), thus requiring increasingly more accurate dynamical models [@cappuccio_report_2020;@asmar_spacecraft_2005;@mazarico_europa_2023;@cappuccio_analysis_2025]. 
One of the major limitations of current dynamical modelling of deep-space probes consists in the complex interaction between the spacecraft shape and the atmosphere, and with radiative forces (solar radiation pressure, albedo, thermal infrared radiation). We developed the pyRTX software package to address this limitation.
Leveraging the ray-tracing technique, originally developed in computer graphics, instead of relying on simplified macro-models (flat plates, cylinders, etc) pyRTX is able to use the actual 3D shape of the spacecraft (provided as, for example, an .obj file), to compute several non-gravitational accelerations. 

pyRTX addresses the gap in open source solutions for a comprehensive modelling of several, important, non-gravitational forces. Being built around the NAIF SPICE astrodynamic library [@acton_ancillary_1996] and its Python wrapper [@annex_spiceypy_2020], pyRTX is intended to be a plug-in tool that can be used in existing OD codes, and applications.

# Functionality

In this section we describe the main functionalities of the pyRTX software. All of these functionalities are discussed in the set of example scripts and notebooks included in the code distribution. Our companion paper [@zurria_refining_2026] discusses an actual application of the pyRTX library for ameliorating the OD of NASA's Lunar Reconnaissance Orbiter. 

- *3D Spacecraft Modeling*: pyRTX relies on detailed 3D mesh models of the spacecraft. This approach inherently allows the software to take into account self-shadowing (where spacecraft components block light or flux from reaching others) and multiple reflections in the computation of accelerations. Users can specify optical properties for each mesh face, enabling the simulation of complex surface interactions such as specular and diffuse (Lambertian) reflections.

- *Solar and Planetary Radiation Pressure Modeling*: pyRTX computes the accelerations due to radiation pressure by casting rays from a pixel plane representing the incoming flux towards the spacecraft. This unified ray-tracing engine calculates the momentum transfer from direct solar photons as well as radiation reflected (albedo) and emitted (thermal infrared) by planetary bodies.

- *Eclipse and Shadow Function Analysis*: pyRTX can compute precise shadow functions during eclipse transitions. It supports advanced modeling features such as solar limb darkening, where the variation in intensity across the solar disk is accounted for in the flux calculation.

- *Atmospheric Drag*: Ray-tracing is used to compute the effective aerodynamic cross-section of the spacecraft by casting rays from a pixel plane representing the incoming atmospheric flux. A user-defined atmospheric density function can be defined to directly compute the drag acceleration. This flexible interface allows the user to interface the precise cross-section calculation with external atmospheric density models.

- *3D Planetary Modeling*: The software supports the use of complex planetary shape models (e.g., topography from digital elevation models, DEMs) and spatially variable maps for albedo, emissivity, and surface temperature. This capability refines not only the planetary radiation fluxes but also the computation of eclipse transitions by accounting for local topography.

- *Lookup Table (LUT) Generation*: Toe nable efficient integration with OD software, pyRTXc anpre-compute accelerations over a grid of incident directions. This feature supports spacecraft with articulating components (e.g., solar arrays), allowing users to generate comprehensive LUTs that mapa cceleration vectors to specific spacecraft orientations ands trongly reduce computation time.

# Acknowledgements

Work by G.C. was supported by NASA under award No. 80GSFC24M0006.

# References