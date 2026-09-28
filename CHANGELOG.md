# Changelog

## 2026-09-28

- Checked the solver against the lithostatic column (runs made 2026-09-24, committed today), the one default case with a
  closed form. Five seeds at the README configuration: the top face carries a
  compression of −0.75 ± 0.32 where a free surface should read 0, top displacement
  −0.98 ± 0.26 against −0.417, and the seed with the smallest loss is the worst.
  Cause: no traction condition on the free surface, so the stress level is
  undetermined. New README section, `results/lithostatic_check.json`,
  `tests/test_readme_numbers.py` (8 tests).
