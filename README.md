# OreForge

**Ore-body modelling with a physics-informed neural network for tectonic deformation.**

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
height in every run, close to the 1.0 the equilibrium equation demands. What differs
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
base.

`tests/test_readme_numbers.py` pins every number in this section to the results file.

## Status

Working application, in active development. Not open source: the Tecto-PINN solver is the
subject of pending provisional patent applications. Source available for review under NDA.

Documentation, figures and result files in this repository are released under CC BY 4.0.
The source code is held in a private repository, all rights reserved.

## History of this page

The code lives in a private repository. This public repository holds the page you are
reading, three screenshots, the result file of the lithostatic check above and the tests
that pin this page to it (see `CHANGELOG.md`).

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
