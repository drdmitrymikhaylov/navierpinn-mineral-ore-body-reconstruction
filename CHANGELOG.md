# Changelog

## 2026-10-21

- The lithostatic column again, for 25 seeds (seeds 0–4 reproduce the September runs to
  every stored digit). The top-face compression is there in 25 of 25 seeds, 0.77 ± 0.21
  (0.23 to 1.01); the network's own top displacement follows from its fitted stress
  profile as a laterally confined column within 2.4 % ± 1.7, so the "overburden"
  reading is mechanical, not a name. The gradient of σ_zz is 0.79 to 0.88 in every seed,
  never the 1.0 a column with traction-free sides needs. Confinement holds at the base
  (σ_xx / σ_zz 0.332 ± 0.001) and scatters at the top (0.21 ± 0.13), and the three seeds
  with the smallest offset are the three least confined — correction: seed 4, "within 4 %
  at the base", has the top stress state furthest from the column. The final loss
  correlates with the error at Spearman ρ = −0.69 / −0.72: the loss ranks the runs
  backwards. New README section, `results/offset_check.json`,
  `tests/test_offset_check.py` (8 tests).

## 2026-10-19

- LICENSE now holds the standard CC BY 4.0 legal text, so GitHub recognises the licence (it showed "Other" before). The terms are unchanged; the note on what each licence covers moved to `NOTICE.md`.

## 2026-09-28

- Checked the solver against the lithostatic column (runs made 2026-09-24, committed today), the one default case with a
  closed form. Five seeds at the README configuration: the top face carries a
  compression of −0.75 ± 0.32 where a free surface should read 0, top displacement
  −0.98 ± 0.26 against −0.417, and the seed with the smallest loss is the worst.
  Cause: no traction condition on the free surface, so the stress level is
  undetermined. New README section, `results/lithostatic_check.json`,
  `tests/test_readme_numbers.py` (8 tests).
