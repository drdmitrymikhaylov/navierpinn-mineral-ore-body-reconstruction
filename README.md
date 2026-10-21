# OreForge

**Ore-body modelling with a physics-informed neural network for tectonic deformation.**

## Field record

OreForge grew out of exploration work, not out of a benchmark. The same question came from
programmes in East Africa, West Africa, Central Asia and the Andes: a handful of
drillholes are in, and nobody can say where the fault moved the ore body between them.
Geostatistics interpolates grade. It does not know that rock has to stay in mechanical
equilibrium. I put that equation into the model. The seismic side of the same research
line is protected by granted US patent 12,693,443 B1 (full-waveform inversion via
generative deep learning, 2026). The solver on this page is covered by the provisional
applications mentioned below.

## Overview

OreForge is a desktop application for mineral exploration geology. It builds a 3D block
model of an ore body, deforms it with faults and folds the way real structural geology
does, and solves the resulting elastic stress field with a physics-informed neural
network. All of this happens inside one interactive 3D workspace.

![OreForge main window](docs/images/01_orebody.png)

<sub>Synthetic ore body coloured by grade above a 1.0 g/t cutoff, with twelve exploration
drillholes sampling the same field.</sub>

---

> ### Source code is not public
>
> OreForge is under active development, and provisional patent applications covering the
> Tecto-PINN solver are pending. The code repository is private. **The source is available
> for technical review under NDA**, contact details are at the end of this page.
>
> This page documents what the system does, how it is built, and what it produces.

---

### What the application does

| | |
|---|---|
| **Block modelling** | Regular 3D block model (40 x 40 x 20 = 32,000 blocks in the demo) with grade, density and rock-type attributes. Grade-cutoff thresholding, and per-block inspection by clicking in the viewport. |
| **Drillholes** | Collar, orientation and assay data model with down-hole interval sampling. Rendered as coloured tubes, so assays and the block model can be read against each other. |
| **Structural geology** | Fault planes defined by strike, dip and slip; folds by wavelength and amplitude. Structures deform the grade field itself, so the block model shows a genuinely faulted and folded ore body rather than a decorative overlay. |
| **Tecto-PINN** | A physics-informed neural network that solves the linear-elasticity equilibrium equations under gravity with a fixed base. It returns displacement and von Mises stress over the model volume, and is trained live, in-app, with a streaming loss curve. |
| **Results overlay** | Any PINN field (von Mises stress, displacement magnitude, vertical displacement) is written back onto the block model and rendered in the same 3D scene as the geology. |

### Related repositories

- [**making-pinns-work**](https://github.com/drdmitrymikhaylov/making-pinns-work), a course
  on why physics-informed neural networks fail to converge, with runnable notebooks. The
  method behind the solver above, taught openly.
- [**cough-spectrograms**](https://github.com/drdmitrymikhaylov/cough-spectrograms), the
  same measurement discipline applied to acoustics rather than mechanics.

## Architecture

```
oreforge/
├── core/           geology - no Qt, no torch
│   ├── block_model.py    regular 3D grid, attributes, PyVista export
│   ├── drillhole.py      collars, orientations, assay intervals
│   ├── geometry.py       FaultPlane (strike/dip/slip), Fold (wavelength/amplitude)
│   └── datasets.py       synthetic ore bodies and consistent drillhole sampling
├── pinn/           the solver - no UI
│   ├── tecto_pinn.py     network, PDE residual, boundary loss, stress recovery
│   └── trainer.py        QThread worker, live progress and cooperative stop
└── ui/
    ├── viewport3d.py     PyVista/VTK scene, picking, scalar overlays
    ├── main_window.py    menus, toolbar, docks, signal wiring
    └── panels/           project tree, inspector, PINN config + loss plot, log
```

Three layers, one direction of dependency. `core` knows nothing about Qt or torch, `pinn`
knows nothing about the UI, and `ui` composes both. The geology can be scripted headlessly,
and the solver can be swapped or benchmarked without touching the interface.

### Threading

Training runs on a `QThread` so the interface stays live. Two details cost real debugging
time. First, torch's intra-op thread pool has to be pinned before any worker starts, or it
deadlocks when first initialised off the main thread. Second, the second-derivative
autograd graph overflows the 512 KB default stack of a macOS secondary thread, so the
training thread is given 128 MB.

### Stack

Python, PyTorch, PySide6 (Qt 6), PyVista / VTK, NumPy, pandas, Matplotlib.

About 1,400 lines of Python across 21 modules. A headless logic test stubs the OpenGL
viewport and exercises the full window wiring without a GPU context: project generation,
structural deformation, block picking, scalar switching and all three PINN overlay fields.

## Screens

### Faulted and folded

The same deposit after adding a fault (strike 45°, dip 72°, slip 25 m) and a fold
(wavelength 420 m, amplitude 14 m). The grade field is displaced across the fault plane.
The offset is in the geology, not painted on top of it.

![Faulted and folded ore body](docs/images/02_faulted_folded.png)

### Tecto-PINN stress field

The trained network's von Mises stress field overlaid on the block model, with the live
training loss in the control dock. 300 epochs, 2,048 collocation points, final loss
7.5e-4, 12 seconds on a laptop CPU.

![Tecto-PINN von Mises overlay](docs/images/03_pinn_vonmises.png)

## Checking the solver against the lithostatic column

*Computed 2026-09-24, added 2026-09-28.* The screenshot above shows a smooth stress field and a small loss.
Neither says whether the field is right. The one case with a closed-form answer is the
solver's own default problem: a homogeneous block under its own weight, fixed at the
base. In the normalised unit cube the solver works in (ρg = 1, E = 1, ν = 0.25, so
λ = μ = 0.4), the laterally confined column gives

| quantity | closed form | at the base | at the top |
|---|---|---|---|
| vertical stress σ_zz | −(1 − z) | −1 | 0 (free surface) |
| horizontal stress σ_xx = σ_yy | σ_zz / 3 | −0.333 | 0 |
| von Mises | (2/3)(1 − z) | 0.667 | 0 |
| vertical displacement u_z | −(z − z²/2) / 1.2 | 0 | −0.417 |

I trained the solver headlessly with the configuration quoted under the screenshot
(300 epochs, 2,048 collocation points) for five seeds and read the fields on 400 random
points per horizontal level (`results/lithostatic_check.json`). Seed 0 reproduces the
numbers under the screenshot: final loss 7.5 × 10⁻⁴, 11.9 s. The fields do not
reproduce the column:

| | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean ± sd | closed form |
|---|---|---|---|---|---|---|---|
| final loss | 7.5e-4 | 3.6e-3 | 1.1e-3 | 3.0e-3 | 2.1e-3 | | |
| σ_zz at the top | −1.02 | −0.86 | −0.97 | −0.65 | −0.24 | **−0.75 ± 0.32** | **0** |
| σ_zz at the base | −1.89 | −1.66 | −1.82 | −1.44 | −1.03 | −1.57 ± 0.35 | −1 |
| von Mises at the base | 1.26 | 1.11 | 1.22 | 0.97 | 0.72 | **1.06 ± 0.22** | **0.667** |
| u_z at the top | −1.22 | −1.07 | −1.17 | −0.88 | −0.57 | **−0.98 ± 0.26** | **−0.417** |

sd is the sample sd over the five seeds. Within any one seed the stress is uniform over
each horizontal face (sd over the face 0.004–0.036), and σ_zz rises by 0.80–0.88 per unit
height in every run, short of the 1.0 that equilibrium demands of a column with
traction-free sides (12–21 % short in all 25 seeds of the next section). What differs
between seeds is a constant offset of the whole profile: the top face carries a
compression of 0.24 to 1.02 that should be zero, as if an unmodelled overburden of that
many domain heights sat on the model, and the base stress, the top displacement and the
von Mises overlay scale with it. The best seed (4) is within 4 % of the column at the
base and still 1.4× on top displacement; the seed with the smallest loss (0) is the
worst, at 1.9× the base stress and 2.9× the top displacement.

The cause is the problem statement, not the optimiser. The solver constrains the
interior equation and the fixed base, and nothing on the top or lateral faces. Equilibrium
with gravity fixes the *gradient* of σ_zz; the *level* is fixed only by a traction
condition on the free surface, which is absent, so any uniform compression added to the
answer satisfies everything that is being minimised. The network picks a level by
initialisation, which is why the same settings give a 2.1× spread in top displacement
across seeds while the loss stays between 7 × 10⁻⁴ and 3.6 × 10⁻³ throughout, and why
the smallest loss belongs to the worst run. Until a free-surface condition is added, the
von Mises overlay in the screenshot is correct in shape (linear in depth, uniform across
each level) and wrong in magnitude by a seed-dependent factor between 1.03 and 1.9 at the
base. That holds at the base; near the top of the seeds with the smallest offset it is
wrong in shape as well (next section).

`tests/test_readme_numbers.py` pins every number in this section to the results file.

## The column again, 25 seeds

*Computed and added 2026-10-21.* Five seeds left two things open: how the offset is
distributed, and whether "an unmodelled overburden" is an explanation or a name. The same
configuration was trained for 25 seeds (0–24; seeds 0–4 reproduce the table above to every
stored digit; 298 s in all), and the σ_zz profile of every seed was fitted as a straight
line through its five levels (largest residual 0.008). `results/offset_check.json`.

| | 5 seeds | 25 seeds | closed form |
|---|---|---|---|
| compression at the top face | 0.75 ± 0.32 | 0.77 ± 0.21 (0.23 to 1.01, median 0.84) | 0 |
| σ_zz at the base | −1.57 ± 0.35 | −1.59 ± 0.22 (−1.89 to −1.03) | −1 |
| von Mises at the base | 1.06 ± 0.22 | 1.07 ± 0.14 (0.72 to 1.26) | 0.667 |
| u_z at the top | −0.98 ± 0.26 | −1.00 ± 0.17 (−1.22 to −0.57) | −0.417 |
| gradient of σ_zz with height | 0.80 to 0.88 | 0.82 ± 0.03 (0.79 to 0.88) | 1 |

**The offset is always there.** 25 of 25 seeds put the top face in compression. Seed 4's
0.23 is the smallest of the 25 (then 0.25 and 0.41; the other 22 sit between 0.64 and 1.01),
so the five-seed sd of 0.32 was inflated by the one seed the sample happened to hold; with
25 it is 0.21 and the median 0.84.

**The overburden reading holds, with a second term.** If the column is laterally confined,
its displacement is fixed by its own stress profile and nothing else: from the fitted
σ_zz = a + b z, u_z(z) = (a z + b z²/2) / 1.2. The network's u_z at the top agrees with
that prediction within 2.4 % ± 1.7 (0.6 to 8.3 % over the 25 seeds) — the displacement is
what the stress says it should be in a confined column, so the offset is the mechanical
overburden it was called, not only a name for the number. Its size alone *over*-explains the
excess top displacement (110 % ± 5, 105 to 126 %), because the second error works the other
way: the gradient of σ_zz is 0.787 to 0.875 per unit height in every seed, never 1 — a
column with traction-free sides must gain exactly one unit of stress per unit of depth, so
either the lateral faces carry 12–21 % of the weight in shear (nothing on them forbids it)
or the interior equation is not met to that accuracy; a face mean cannot tell the two
apart. Offset and short gradient together predict 95 % ± 6 of the excess (71 to 99 %).

**Confinement weakens with height, and the three "best" seeds are the least confined.**

| z | σ_zz, 25 seeds | σ_xx / σ_zz (closed form 0.333) | von Mises / \|σ_zz\| (closed form 0.667) |
|---|---|---|---|
| 0 | −1.59 ± 0.22 | 0.332 ± 0.001 (0.33 to 0.33) | 0.674 ± 0.008 (0.67 to 0.70) |
| 0.25 | −1.39 ± 0.22 | 0.316 ± 0.009 (0.29 to 0.33) | 0.693 ± 0.018 (0.68 to 0.75) |
| 0.5 | −1.18 ± 0.21 | 0.294 ± 0.024 (0.21 to 0.32) | 0.719 ± 0.037 (0.69 to 0.84) |
| 0.75 | −0.97 ± 0.21 | 0.263 ± 0.052 (0.07 to 0.31) | 0.757 ± 0.075 (0.70 to 1.01) |
| 1 | −0.77 ± 0.21 | 0.210 ± 0.126 (−0.28 to 0.30) | 0.828 ± 0.189 (0.71 to 1.49) |

At the base every seed is the confined column (σ_xx / σ_zz 0.331 to 0.333). The ratio
drifts off with height and scatters at the top, where it runs from −0.28 to 0.30 and the
von Mises stress is 0.71 to 1.49 times what the vertical stress implies. The lateral
displacement (mean |u_x| or |u_y| over a face, at the level where it is largest) is 1 to
36 % of the top vertical displacement — above 10 % in 12 seeds, above 20 % in 5 — and the
seeds with more of it are the seeds where the confined-column prediction of u_z is worse
(Spearman ρ = 0.59, p = 0.002). The three smallest offsets belong to seeds 4, 10 and 8
(0.23, 0.25, 0.41), and they are the three least confined columns: lateral displacement
26 / 36 / 27 % of vertical, σ_xx / σ_zz at the top −0.28 / −0.02 / 0.01, von Mises at the
top 1.49 / 1.36 / 1.01 times |σ_zz| against the column's 0.667. They lower the top
compression by letting the block bulge sideways, not by finding the free surface.
**Correction** to the section above: seed 4, "within 4 % of the column at the base", has
the right base stress and, of all 25 seeds, the top stress state furthest from the column;
there the von Mises overlay is wrong in shape, not only in level.

**The loss ranks the runs backwards.** Over 25 seeds the final loss correlates with the
top compression at Spearman ρ = −0.69 (p = 1.3 × 10⁻⁴) and with the top-displacement error
at ρ = −0.72 (p = 5 × 10⁻⁵): the five lowest-loss seeds carry offsets of 0.84 to 1.01
(mean 0.94), the five highest-loss 0.41 to 0.85 (mean 0.64). "The seed with the smallest
loss is the worst" is the rule, not a five-seed accident: a run that keeps the column
confined meets the interior equation better and carries a larger overburden (lateral
displacement against offset ρ = −0.73, against loss ρ = +0.46, p = 0.02), and the loss
rewards the first while it cannot see the second. Picking the run with the lowest loss,
which is what a user watching the loss curve in the dock will do, picks the largest error.

*What this does not show.* One configuration (300 epochs, 2,048 collocation points) and
one problem; the fix itself — a traction condition on the top face — has not been run, so
nothing here says how much of the offset it removes or whether the short gradient survives
it; the shear on the lateral faces was not read, only inferred from the face-mean
gradient; the seed sets the solver's own seed parameter, and the 25 runs share the
sampling scheme.

`tests/test_offset_check.py` pins every number in this section to the results file (and
seeds 0–4 of it to `results/lithostatic_check.json`).

## Status

Working application, in active development. Not open source: the Tecto-PINN solver is the
subject of pending provisional patent applications. Source available for review under NDA.

Documentation, figures and result files in this repository are released under CC BY 4.0.
The source code is held in a private repository, all rights reserved.

## History of this page

The code lives in a private repository. This public repository holds the page you are
reading, three screenshots, the two result files of the lithostatic checks above and the
tests that pin this page to them (see `CHANGELOG.md`).

Before the page went public I removed the paragraph that described the solver physics in
detail. Three shorter mentions of linear elasticity remain, in the feature table and the
architecture tree, and I left them in knowingly.

The screenshots are real. I produced them by scripting the running Qt application and
saving the frames at reduced size. Everything in them is synthetic data from
`oreforge/core/datasets.py`; no material from any client project appears on this page.

Taking the screenshots turned up one small bug. Cell picking was leaving PyVista's
"Press R to toggle selection tool" banner across the whole 3D viewport.
`enable_cell_picking` now runs with `show_message=False`, and the banner is gone from the
images above.

## Contact

**Prof. Dr. Dmitry Mikhaylov**, Abu Dhabi, UAE

[LinkedIn](https://www.linkedin.com/in/dmitry-mikhaylov) |
[ORCID](https://orcid.org/0009-0009-2108-6820) |
[Substack](https://dmitrymikhaylov.substack.com)
