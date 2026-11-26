# Full neural panel NPIV

This is the only retained NPIV implementation. Both stages are neural networks:
a Gaussian-mixture treatment density and a tanh structural response. The panel
has 101 communes observed in 2008–2017, eleven raw origin-share instruments,
age/male-share controls, year inputs, and additive commune/year nuisance effects.

Run commands from the repository root after installing `requirements-neural.txt`
in a Python environment. The raw data path is
`rep_data/Replication/data/analysis/IV_panel.dta`; download instructions are in
the repository's root `requirements.txt`.

```sh
python npiv/neural/run_full_neural.py --check-only --output /tmp/neural-preflight
python npiv/neural/verify_full_neural.py
```

The first command checks numerical identities without fitting the panel. The
second reconstructs saved predictions, held-out losses, derivatives, and fixed
effects from the original data and networks.

- `run_full_neural.py`: fit and select both stages. Use a new `--output` directory
  for an intentional rerun; the default is the saved `full_neural/` reference.
- `panel.py`: data aggregation, raw-share inputs, commune folds and fixed effects.
- `outcome_network.py`: response network, analytic derivatives and time summaries.
- `reference_design.json` and `reference_profiles/`: fixed sample/fold/profile
  design, independent of any removed estimator.
- `bootstrap_full_neural.py`: conditional commune bootstrap of both stages.
  Its model source remains `full_neural/`; `--output` changes only the inference
  destination. Existing 399-draw results do not need to be rerun for review.
- `full_neural/`: saved fits and diagnostics; `inference/summary.csv` contains
  standard errors, centered-bootstrap p-values and intervals.
- `full_neural/refactor_validation.json`: numerical checks after extracting this
  implementation and deleting the other methods. Original fitting hashes remain
  in the manifests; `refactor_compatibility` records the new code separately.
- `write_full_neural_memo.py`: rebuild `NPIV.tex` from saved results and editable
  memo prose. Compile from the repository root with `latexmk -lualatex NPIV.tex`;
  `.latexmkrc` writes the PDF and build intermediates to `aux/`. The final memo
  is `aux/NPIV.pdf`. The original template requires the external course style
  `../../../adenc.sty`.

See `CLAUDE.md` in the repository root for complete settings, inference limits,
and the unimplemented age/education/year heterogeneity roadmap. No other NPIV
implementation or result directory is needed by this code.
