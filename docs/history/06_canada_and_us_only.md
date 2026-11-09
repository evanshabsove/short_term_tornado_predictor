# Country mask, Canadian coverage, US-only baseline

> Archived verbatim from CLAUDE.md (as of commit 6d2eaed, 2026-11-08) when CLAUDE.md was compacted.  Wording, numbers and any "see above/below" references are as originally written; read them as pointers into the sections of CLAUDE.md that now live in docs/history/ (see docs/history/README.md for the map). Where a later section supersedes an earlier one, the original text says so.

### Country mask and Canadian coverage (Canada-transfer groundwork, not a modeling change)

`scripts/08_canada_transfer/build_country_mask.py` classifies every coarse-grid
cell by country, and quantifies how much of Canada the HRRR domain covers.
Motivated by the long-term goal of extending to Canada (see memory: forecaster-aid
tool, then Canada transfer). Output: `data/processed/hrrr_coarse_grid_stride13_countries.nc`;
`scripts/08_canada_transfer/plot_country_mask.py` plots it (`outputs/country_mask.png`,
grid-space, north-up); `tests/test_country_mask.py` spot-checks it (skipped if the
gitignored mask is not built). Does not touch the grid, labels, features or any model.

- **Method:** Natural Earth 1:10m admin-0 countries, **minus the 1:10m lakes layer**, in
  the grid's own LCC meters. Cell-center point-in-polygon gives `country`
  (0 water, 1 US, 2 CA, 3 MX, 4 other land); exact LCC-square intersection gives
  `frac_{us,ca,mx,other_land,water}`; `straddles_border` / `straddles_coast` flag cells
  with >= 2 land classes / land+water each above 1% of the cell (`--min-frac`).
- **A real bug was caught and fixed:** the first build assumed Natural Earth's admin-0
  polygons exclude the Great Lakes. They do not -- the border runs through the lake
  middles -- so lake cells came out as US/Canada with water fraction 0. Caught by looking
  at the plot (no blue lakes) and confirmed by probing mid-lake cells; fixed by subtracting
  the lakes layer. Covered by tests so it cannot silently return.
- **Counts (81x138 = 11,178 cells, after the fix):** water 4,121 (36.9%), US 5,148,
  CA 1,243 (11.1%), MX 659, other land 7. 192 border-straddling and 1,305 coast-straddling
  cells. 1,442 cells have any Canadian area (~1.91M km^2). Canadian cells span ~42.1-52.3N.
  Before the fix these were water 3,895 / US 5,276 / CA 1,339 / MX 661.
- **Northern coverage gap:** the domain's top row of cell centers is 47.6-52.3N (top edge
  ~52.5N at its highest; it dips in the east because of LCC curvature), so everything north
  of ~52N is outside. Share of each region's area inside the grid extent (not lake-adjusted):
  PEI/New Brunswick 100%, Ontario 71%, Nova Scotia 65%, Saskatchewan 32%, Quebec 28.5%,
  Manitoba 25%, Alberta 15%, BC 15%; Yukon, Nunavut, NWT, Newfoundland & Labrador 0%
  (all of Canada 16.9%). **This is area coverage, not tornado-weighted coverage** -- no
  Canadian tornado dataset is in the repo yet, so how much of Canada's actual tornado
  climatology falls inside is unmeasured.
- **Alberta caveat (from memory, NOT verified against data):** Alberta's tornado activity
  is concentrated in the central part of the province (Calgary-Red Deer-Edmonton, ~51-53.5N).
  Calgary is inside the domain, Red Deer is at the top edge, Edmonton (53.5N) is outside --
  so the Alberta gap likely includes real tornado country, and cells within a few rows of the
  top edge also have less spatial context for the U-Net's receptive field.
- **Confirmed: Canadian cells are essentially always labeled 0.** The SPC CSV
  (`spc_tornado_reports_2014_2025.csv`, 15,294 reports) contains only US states/territories
  (no Canadian province codes; its `CA` is California). Checked against
  `tornado_labels_2014_2025.csv` (147,440 rows): of 3,237 unique positive cells, 3,122 are US,
  111 water, 3 MX, and **1 of the 1,243 Canadian cells** (24 label rows, a border-crossing US
  track) -- the other 1,242 are never positive, whether or not a Canadian tornado occurred
  there. That is label noise on ~11% of the grid. Magnitude of the effect on metrics/training
  is NOT yet measured (Canadian tornadoes are probably a small fraction of the ~1,400/yr US
  count, but that figure is from memory). Candidate fixes (not done): evaluation-only rescoring
  with Canadian cells masked out (cheap, no retrain), masked-loss retraining (needs >=3 seeds
  per the seed-variance rules), or Northern Tornadoes Project / Environment Canada labels.
- **Next steps, not done:** quantify NTP/ECCC tornado counts inside vs. outside the grid and
  by distance to the top edge; investigate Canadian NWP (HRDPS, RRFS) archive depth before any
  model switch. An inquiry about NTP data access was sent to the Northern Tornadoes Project
  (no reply recorded here).

### US-only baseline (mask non-US cells from loss and metrics): Stage A and Stage B done

Because SPC has no Canadian/Mexican labels, those cells (1,909 of 11,178: CA 1,243, MX 659, other land 7) are always labeled 0
(see "Country mask" above). `cell_mask.py` defines the **US-trusted mask** (`country` in {water, US}, 9,269 cells kept); `FocalLoss`
(`mask=`), `training.train_model(cell_mask=)`, `evaluate.get_proba_labels/evaluate_model/evaluate_checkpoint(cell_mask=)` and
`cross_validate_year.py --cell-mask us_trusted` use it. Default is off and bit-for-bit unchanged (206 tests pass, incl.
`tests/test_cell_mask.py`). Design choices: masked loss is divided by the TOTAL cell count (not the kept count) so loss scale/effective LR
match the unmasked baseline; a map whose only positive is in an excluded cell counts as quiet for `QuietMapDownsampler`
(`training.masked_is_active`); **inputs keep the full grid** (the CNN still sees weather across the border); mask provenance
(`cell_mask_name`, kept/total cells) is saved in each checkpoint's `training_hyperparameters`.

- **Stage A (evaluation-only rescore, `scripts/08_canada_transfer/rescore_with_mask.py`, results `models/us_only_study/stage_a_rescore.json`):**
  existing models (trained with the UNMASKED loss), cached scores, metric-only exclusion. All-True mask reproduces the unmasked numbers exactly.
  Masked AUC-PR >= unmasked in 12/12 folds for both the old and v4 12-fold models and 18/18 EMA-study single models, but the gain is tiny:
  **old +0.4%, v4 +0.7%, EMA-study singles +0.4-0.5%** (6-seed ensembles +0.0-0.2%); best-F1 +0.1-0.3%. Pre-registered predictions held (>=9/12 improved;
  gain in the 0-10% range, well inside the ~19% seed noise; best-F1 moves less than AUC-PR). **Conclusion: scoring Canadian/Mexican cells as
  negatives barely moves current metrics** -- the model already gives those cells low probability. Mean predicted p (v4, pooled): US 0.0128, CA 0.0027,
  MX 0.0038 (CA peaks in July, 0.0042, consistent with the summer severe season). This does NOT show the label noise is harmless for training (the models
  still learned from it) or say anything about Canadian skill (no Canadian labels exist to test it).
- **Stage B (masked-loss retrain; `scripts/08_canada_transfer/run_us_only_study.sh`, evaluator `us_only_eval.py`, results `models/us_only_study/us_only_eval.json`):**
  EMA 0.9999, seeds 10-12, folds 2016/2020/2024, Canada/Mexico/other-land cells excluded from loss and metrics, inputs keep the full grid. Scored on
  US-trusted cells only against the unmasked-trained averaged models (seeds 10-15; 20 three-member baseline ensembles). **Pre-registered rule outcome:
  no detectable effect.** 3-member ensemble AUC-PR vs. the mean of the 20 baseline triples: 2016 0.0398 vs 0.0408 (-2.4%, 35th pct), 2020 0.0623 vs 0.0578
  (+7.9%, above all 20), 2024 0.0718 vs 0.0722 (-0.5%, 45th pct); pooled +1.7%, positive in 1/3 folds ("helps" needed >= +8% and 2/3 folds). Against the matched
  triple (seeds 10-12) the paired day-bootstrap intervals include 0 in all three folds. Best-F1 and precision/recall at best-F1 are within a few percent
  (2020 recall 0.221 vs 0.226; 2016 recall 0.187 vs 0.158 but precision 0.080 vs 0.085). Single-model AUC-PR spread is large as usual (2016 US-only seeds 0.0429/0.0205/0.0277).
  **Conclusion: masking unlabeled cells from the training loss gives no detectable change in US skill; this US-only model is a valid clean baseline, not a better one.**
  Descriptive: mean predicted p over Canadian cells is HIGHER with masked training (2016/2020/2024: 0.0037/0.0060/0.0042 vs 0.0016/0.0014/0.0009 for unmasked training; MX similar),
  consistent with unmasked training having pushed probability down over cells it was always told were negative, and masked models no longer being penalized there. US cells
  are within +/-10% either way (0.0111/0.0111/0.0104 vs 0.0124/0.0105/0.0113). This is the expected direction, not a measure of Canadian skill (no Canadian labels exist).
  An earlier launch of this run failed instantly (bare `python` hit a broken pyenv shim because the venv was not activated; `tee` hid the exit code); it was relaunched with
  the venv active. Stage C (add NTP/ECCC Canadian labels, compare against this US-only baseline) is a separate, not-yet-planned ticket.


