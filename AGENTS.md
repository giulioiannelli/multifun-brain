# Agent Guide for `multifun-brain`

This repository contains utilities and notebooks for constructing, analysing, and visualising hierarchical modular brain networks with an emphasis on fMRI correlation matrices, band-limited filtering, and Laplacian renormalisation group (LRG) style multiscale analysis.

## Repository scope
- **Package modules (`multifunbrain/`)**: core signal utilities, correlation-network helpers, Laplacian diffusion/LRG routines, graph-thresholding tools, network generation, and visualisation helpers.
- **Notebooks (`notebooks/`)**: end-to-end exploratory workflows for computing band-specific correlation matrices, running diffusion-based clustering, and comparing fMRI contrasts or scales.
- **Documentation (`docs/`) and config**: Markdown guides, changelog, packaging metadata, and contribution guidelines.

## Agent instructions
- Prefer reusing existing utilities from `multifunbrain.analysis`, `multifunbrain.core`, and `multifunbrain.visualization` instead of duplicating logic.
- Keep notebooks self-contained and portable: use relative `Path` operations, avoid hard-coded user-specific directories, and describe inputs/outputs in markdown cells.
- When adding analytical steps (e.g., noise filtering, clustering comparisons), provide concise docstrings or markdown notes that link the step back to the multiscale/LRG workflow.
- Do not add large data files to the repository. Place outputs in a `results/` directory created by the notebook if needed.
- Follow existing formatting conventions (PEP8 in code cells, 88-character line guidance) and keep variable names descriptive for fMRI/graph contexts.
