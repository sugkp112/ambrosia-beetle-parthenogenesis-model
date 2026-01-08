# Ambrosia Beetle Parthenogenesis Model

Simulation code for studying the emergence of **facultative parthenogenesis**
under mate-finding limitation in spatially structured populations.

The model was developed to explain **single-female colonization in ambrosia beetles**
and demonstrates how asexual reproduction can arise at population fronts
without assuming any fitness advantage or strategic choice.


# Model Description

This is a mechanistic individual-based model integrating:

- Spatial population structure on a 2D grid
- Local encounter constraints (maximum two individuals per cell)
- Mate-finding Allee effects on sexual reproduction

Reproductive mode is determined solely by local encounters:

- Single individuals reproduce asexually
- Paired individuals attempt sexual reproduction
- Sexual success depends on local density

Asexual reproduction emerges as a demographic consequence of mate-finding failure,
not as an adaptive strategy.

# Main File
encounter_limited_model.py  
Generates all simulation results and figures reported in the manuscript,
including temporal dynamics, spatial and radial profiles,
phase-specific analyses, and sensitivity analyses of Allee thresholds.

# Run
python encounter_limited_model.py
All figures are generated automatically.

# Requirements
numpy
scipy
matplotlib
seaborn

# Citation
Jiang, Z.-R. (2026).
Mechanistic model of encounter-limited reproduction and facultative parthenogenesis
in ambrosia beetles.

GitHub: https://github.com/sugkp112/ambrosia-beetle-parthenogenesis-model

# Contact
For questions, discussion, or collaboration:

Email: ziru.jiang@gmail.com
