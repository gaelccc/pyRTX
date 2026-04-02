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
date: 13 February 2026
bibliography: paper.bib

---

# Summary

With the constant improvement of radiometric tracking systems, inaccuracies in non-gravitational force modeling have become one of the limiting factors in precise orbit determination for deep space missions and the scientific products that depend on it. A major source of modeling error arises from the complex 3D shape of spacecraft. While fast and reliable analytical models exist for simple geometries (e.g. spheres or flat plates), they are generally not available for more realistic spacecraft representations. We present pyRTX, a software package that addresses this limitation by leveraging ray-tracing techniques to model the interaction between the space environment (radiation and atmosphere) and detailed spacecraft geometries. By operating directly on accurate 3D mesh models, pyRTX enables high-fidelity computation of non-gravitational accelerations. The software is specifically geared towards planetary orbit determination where inaccuracies of the model often cannot be overcome with continuous tracking as it is routinely done for Earth-orbiting spacecraft.




# Statement of need

Accurate reconstruction of spacecraft trajectories is essential for several scientific applications, including the determination of the gravity field of Solar System bodies. These tasks are accomplished by solving the orbit determination (OD) problem [@tapley_statistical_2004; @milani_theory_2009]. The solution is retrieved by minimizing the difference between observed data and predictions. This is accomplished through an iterative adjustment of a dynamical model (a set of differential equations) describing the spacecraft motion. Systematic errors in the dynamical model directly translate into biases in the estimated trajectory and derived geophysical parameters. In recent years, improvements in radiometric tracking systems have significantly increased measurement precision [@cappuccio_report_2020; @asmar_spacecraft_2005; @mazarico_europa_2023; @cappuccio_analysis_2025], placing tighter requirements on the accuracy of dynamical models.

One of the main limitation in the dynamical modeling of deep space probes, especially in the inner Solar System, is the simplified treatment of non-gravitational forces, such as radiation pressure or atmospheric drag. This occurs as photons or atmospheric particles interact with the probe's surfaces, leading to a momentum exchange and, thus, an acceleration. Existing approaches typically represent the spacecraft shape with simplified macro-models (e.g. flat plates or cylinders). While fast and efficient, these methods cannot fully capture effects such as self-shadowing (when parts of the spacecraft are shielded from the incoming flux by other structural elements) and multiple reflections (reflected particles which impact subsequent surfaces) [Mazarico et al. 2014; Li et al. 2014].

The pyRTX software package addresses this gap by implementing a ray-tracing-based approach for computing non-gravitational accelerations using realistic spacecraft geometries. Built around the NAIF SPICE library [@acton_ancillary_1996] and its Python interface [@annex_spiceypy_2020], pyRTX is designed to integrate with existing OD pipelines and provide improved dynamical modeling.

# State of the field

Non-gravitational force modeling in orbit determination has traditionally relied on the simplified “plate model”, which consists in the discretization of the probe as a set of plates or 2D elementary shapes. This approach enables efficient computation but lacks the capability to fully capture the complex interaction between particles and complex spacecraft geometries. Ray-tracing techniques, widely used in computer graphics, have been explored to address this task [Darugna et al. 2018; Kenneally & Schaub 2020; Li et al. 2018]. However, to the best of our knowledge, there is currently no open-source software package that computes solar and planetary radiation pressure and atmospheric drag with a ray-tracing approach, allowing direct integration with standard astrodynamics libraries. pyRTX fills this gap by providing a flexible and open-source tool for high-fidelity non-gravitational force modeling, specifically designed for orbit determination applications. 

# Software design

pyRTX is designed as a modular and extensible library. The software operates on detailed 3D spacecraft models and computes non-gravitational accelerations by simulating the interaction between incoming fluxes and the spacecraft geometry. Built around the NAIF SPICE toolkit, it ensures compatibility with standard astrodynamics pipelines. At its core, pyRTX uses a ray-tracing engine [Reference a pyEmbree?] in which rays are cast from a discretized representation of the incoming flux toward the spacecraft. This approach has the advantage to account for multiple physical effects:

- *3D spacecraft modeling*: The software operates directly on high-resolution 3D mesh models (.obj files), allowing accurate representation of spacecraft geometry. This naturally accounts for self-shadowing and multiple reflections. Optical properties can be assigned for individual mesh elements.

- *Radiation pressure computation*: Solar radiation pressure and planetary albedo and thermal infrared emission are computed by casting rays from a pixel plane representing the incoming flux towards the spacecraft [Ziebart et al. 2004]. Momentum transfer is evaluated through ray-surface interactions, capturing both specular and diffuse reflection components.

- *Eclipse and shadow modeling*: pyRTX computes shadow functions with high fidelity, including partial illumination conditions and solar limb darkening, improving the modeling of eclipse transitions.

- *Atmospheric drag computation*: The effective aerodynamic cross-section is derived via ray tracing, with the possibility to define an atmospheric density model. This allows flexible and accurate drag modeling without relying on simplified geometric assumptions.

- *Planetary surface modeling*: The software supports complex planetary shape models (e.g. digital elevation models) and spatially variable surface properties such as albedo, emissivity, and temperature. This capability refines not only the computation of planetary radiation fluxes but also the eclipse transitions by accounting for local topography.

- *Lookup table generation*: The software can precompute accelerations over grids of incident directions and spacecraft orientations. This enables fast interpolation during the computation of non-gravitational accelerations, strongly reducing the computational times.


# Research impact statement

pyRTX enables a new level of accuracy in non-gravitational force modeling, addressing a major challenge in the orbit determination of deep space missions. By leveraging ray-tracing techniques and realistic spacecraft geometries, the software reduces modeling errors that can otherwise bias scientific results. By providing an open-source and extensible framework, pyRTX supports the scientific community across different missions and institutions.

The package has already been applied to the orbit determination of NASA’s Lunar Reconnaissance Orbiter [@zurria_refining_2026], demonstrating its relevance and impact on the data analysis of a present mission. The software is expected to support future missions where non-gravitational effects are dominant and critical to the science objectives, including those to Mercury, Venus, Earth, the Moon, and Mars.

# Acknowledgements

Work by G.C. was supported by NASA under award No. 80GSFC24M0006.

# AI Usage Disclosure

Generative AI tools were used to support the preparation of the manuscript and the documentation of the the software. The development of the code, including its algorithms, implementation, and testing was carried out without AI assistance. All content produced with the support of AI has been carefully reviewed and verified by the authors to ensure its validity.

# References