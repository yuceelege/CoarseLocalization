# CoarseLocalization

ActiveLocalize is a compact research codebase for **set-based active localization** in a 2D linear system with binary proximity measurements.

This repository contains the code for the paper:

**CoarseLocalization of Unstable Systems with Coarse Information**  
Ege Yuceel, Daniel Liberzon, Sayan Mitra  
arXiv: https://arxiv.org/pdf/2602.06191  
Accepted to appear in the **ACM International Conference on Hybrid Systems: Computation and Control (HSCC 2026)**.

At each step, the algorithm:
- propagates the current feasible state set,
- intersects it with geometry induced by positive measurements,
- updates a landmark uncertainty set,
- and logs convergence metrics (state-set and landmark-set diameters).

## Repository layout

- `main.py` — runs localization experiments and saves per-trial CSV logs.
- `configs.py` — system matrices, region sizes, and experiment constants.
- `tools/utils.py` — geometry helpers (`create_ellipse`, recovery signal, enclosing circle).
- `tools/findsvp.py` — utility for spherical/circular Voronoi point-set design.
- `plotting/plotter.py` — aggregates trials from one experiment and plots state + landmark error curves.
- `plotting/plotter_landmark.py` — landmark-only aggregate plot.
- `results/` — output folder (each experiment goes in `results/<experiment_name>/`).

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run an experiment

```bash
python main.py --num_trials 10 --max_steps 1000 --experiment_name test
```

This creates CSV logs such as:
- `results/test/diam_trial_0.csv`

### 3) Plot aggregated results

```bash
python plotting/plotter.py --experiment_name test --max_index 1000
```

Outputs:
- `state_estimation_error.png`
- `landmark_estimation_error.png`

Landmark-only plot:

```bash
python plotting/plotter_landmark.py --experiment_name test --max_index 1000
```

## Notes

- The code assumes a discrete-time linear system `x_{k+1} = A x_k + B u_k`.
- Measurements are based on distance-to-landmark thresholding with sensor radius.
- Geometry operations rely on `shapely`.

## License

Add your preferred license before publishing (e.g., MIT/BSD-3-Clause).

## Suggested GitHub description (About)

Set-based active localization in 2D linear systems with geometric uncertainty shrinking and landmark-region refinement.