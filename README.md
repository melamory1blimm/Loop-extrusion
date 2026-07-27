# Loop Extrusion and Hi-C Fountain Analysis

---

## Overview

This repository provides computational tools for modeling cohesin-mediated loop extrusion and analyzing fountain-shaped patterns in Hi-C contact maps.

The code includes:

- stochastic simulations of one-sided and two-sided loop extrusion;
- steric interactions between neighboring cohesins and transient obstacles;
- construction of aggregate Hi-C fountains;
- anisotropic Gaussian fitting of fountain geometry;
- numerical fitting of a mechanistic loop-extrusion model;
- estimation of correlations between the two extruded loop arms;
- enhancer and promoter annotation around fountain bases.

---

## Repository Structure

├── extrusion_sim.py

├── fountain_tools_clean.py

├── fountains_exp.py

└── README.md


### `extrusion_sim.py`

Stochastic simulation of loop extrusion by multiple cohesin complexes. The script supports one-sided and two-sided extrusion and accounts for steric blocking between neighboring loops.

### `fountain_tools_clean.py`

Core analysis library containing functions for:

- extraction and aggregation of Hi-C fountain maps;
- observed/expected normalization;
- Gaussian and numerical model fitting;
- calculation of fountain geometry and arm correlations;
- visualization of fitted models;
- enhancer–promoter annotation and analysis.

### `fountains_exp.py`

Analysis workflow used to process experimental Hi-C data, construct aggregate fountains, fit the Gaussian and mechanistic models, and generate the corresponding figures.

Dataset paths and analysis parameters should be adjusted before running the script.

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

The simulation parameters in `extrusion_sim.py` can be modified to compare different extrusion mechanisms, cohesin densities, lifetimes, and obstacle densities.

---
