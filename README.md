# Random Blob Methods

Code accompanying the paper *Blob Methods for Diffusion* by Katy Craig and Claire Murphy.

This repository implements the random batch method and random multirate method for ODE systems
arising from blob methods for the nonlinear Fokker-Planck equation. 
Convergence is assessed by comparing the particle approximation to a reference solution 
in the Wasserstein-2 distance, computed with the [POT](https://pythonot.github.io/) library.

## Structure

- `1d/` — one-dimensional experiments (notebooks) and supporting source (`1d/src/`)
- `2d/` — two-dimensional experiments (notebooks) and supporting source (`2d/src/`),
  including a Navier–Stokes advection case
- `environment.yml` — conda environment specification

Each `src/` directory contains:
- `kernel.py` — the mollifier (blob kernel) and its gradient
- `integrators.py` — time-stepping schemes, including the random batch method (RBM)
  and the random multirate method
- `initial_data.py` — discretization of a continuum initial density into particles
- `error.py` — 2-Wasserstein distance between the particle solution and a reference
  density, via optimal transport

## Setup

```bash
conda env create -f environment.yml
conda activate random-blob-methods
```

## Usage

Launch Jupyter and open any notebook in `1d/` or `2d/`:

```bash
jupyter notebook
```

Each notebook is organized into `Configuration`, `Function Definitions`, `Plot`, and
`Main` sections — adjust parameters in `Configuration` and run all cells.

## Citation

A preprint is forthcoming; citation details will be added here once available.
In the meantime, please cite this repository directly (see "Cite this repository" 
in the sidebar, or use the URL above).
