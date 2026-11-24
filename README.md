# Loop Extrusion and Hi-C Fountain Analysis

---

## Overview

This repository provides computational tools for modeling cohesin-mediated loop extrusion and analyzing fountain-shaped patterns in Hi-C contact maps.

The code includes:
- theoretical analysis of one-sided, two-sided, and Poisson-switching loop extrusion;
- calculation of loop-length distributions and constraints on the switching length;
- stochastic simulations of cohesin extrusion with chromatin barriers and cohesin–cohesin encounters;
- construction of aggregate Hi-C fountains for three vertebrate species;
- anisotropic Gaussian characterization of fountain geometry and bootstrap analysis;
- estimation of correlations between the two extruded arms, including direct empirical estimates.

---

## Repository Structure

├── cohesin_simulator_state_diagnostics.py
├── model_of_switching_poisson_v2.nb
├── fountain_exp.py
├── fountain_tools_clean.py
├── extrusion_sim.py                 # older version
├── fountains_exp.py                 # older version
├── loop fitting.nb                  # older notebook
├── xenopus_tropicalis_enh.py        # separate analysis
└── README.md


### cohesin_simulator_state_diagnostics.py
Stochastic simulation of cohesin-mediated loop extrusion. The script supports one-sided, two-sided, and directional-switching extrusion, accounts for blocking by chromatin barriers and neighboring cohesins, and provides state-resolved diagnostics for comparison with analytical predictions.
### model_of_switching_poisson_v2.nb
Wolfram Mathematica notebook implementing the theoretical model of Poisson switching between active extrusion arms. It calculates loop-length distributions and analyzes the dependence of their peaks on the characteristic switching length.
### fountain_exp.py
Current experimental Hi-C fountain analysis for Oryzias latipes, Xenopus tropicalis, and Danio rerio. The script constructs aggregate fountains, fits anisotropic Gaussian profiles, estimates arm correlations, performs bootstrap and fitting-window analyses, and generates figures and tables for the manuscript. Dataset paths and analysis parameters can be specified when running the script.
### fountain_tools_clean.py
Core analysis library used by fountain_exp.py, containing functions for:
- extraction and aggregation of Hi-C fountain maps;
- observed/expected normalization;
- Gaussian fitting of fountain geometry;
- empirical arm-correlation calculations and bootstrap analysis;
- visualization of experimental fountains and fitted profiles.

---

## Input Data

The main analysis requires:

- Hi-C contact maps in `.cool` or `.mcool` format;
- a table containing fountain genomic coordinates and scores;
- optional enhancer and promoter annotations in BED-, CSV-, or GTF-compatible formats.

Large experimental datasets are not included in the repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/melamory1blimm/Loop-extrusion.git
cd Loop-extrusion
```

Install the main Python dependencies:

```bash
pip install numpy scipy pandas matplotlib cooler cooltools tqdm
```

---

## Usage

The experimental analysis can be run from:

```bash
python fountains_exp.py
```

Before running, specify the paths to the Hi-C maps, fountain coordinates, and genomic annotations.

The simulation parameters can be modified to compare different extrusion mechanisms, cohesin densities, lifetimes, and obstacle densities.

---
