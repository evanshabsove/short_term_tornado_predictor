# Dataset builds: pilot, first scale-up, v1, v2, v4

> Archived verbatim from CLAUDE.md (as of commit 6d2eaed, 2026-11-08) when CLAUDE.md was compacted.  Wording, numbers and any "see above/below" references are as originally written; read them as pointers into the sections of CLAUDE.md that now live in docs/history/ (see docs/history/README.md for the map). Where a later section supersedes an earlier one, the original text says so.

### Consolidated training dataset (pilot + first scale-up)

`src/tornado_predictor/dataset.py` combines feature extraction and
labeling across multiple HRRR runs into one dataset, via
`scripts/02_dataset_build/build_training_dataset.py`.

- **One representative fxx per bin, not full-bin aggregation.** A bin
  spans 4 forecast hours (`time_bins.fxx_values_for_bin`), but each
  sample here uses only the bin's first hour (fxx=0 for bin 0, fxx=4
  for bin 1) as a stand-in for the whole bin — keeps Herbie network
  cost proportional to sample count rather than 4x it. Aggregating all
  4 fxx per bin (spatial pooling already done, plus a further temporal
  reduction) is a possible future refinement, not built here — this was
  an explicit scope tradeoff, not an oversight.
- **Storage keeps (sample, row, col) structure**, one variable per
  field/statistic plus `label` — self-describing, consistent with
  every other saved artifact in this repo. `to_training_arrays(ds)` is
  a separate, explicit function that flattens this into literal
  `(sample, grid, feature)` + `(sample, grid)` numpy arrays for direct
  ML consumption — that flattening is lossy (loses which axis was row
  vs. col) and is deliberately not baked into the storage format.
  "Sample" is the accurate name for the leading axis (each is one
  independent (init_time, bin_index) pair, not a continuous timeline,
  per the every-hourly-run time-binning design) even though a
  consuming ML pipeline will likely treat it as "time".
- **This is a pilot, not a production build.** Demonstrated on 3 hourly
  HRRR runs (2021-12-10 21:00/22:00/23:00 UTC, bracketing the
  2021-12-11 02:54 UTC Quad-State/Mayfield EF4 and the broader outbreak
  that night) — chosen specifically because it's within the labeled
  SPC range *and* produces genuine positive labels (167 positive
  (sample, cell) pairs across all 6 samples), not an all-zero sanity
  check.

### First scale-up (44 train / 6 val samples)

Built via `scripts/02_dataset_build/build_training_dataset.py`, applying `split.py`'s
leakage-safe split (see above) to a **stratified sample of 25 dates**
(2014–2025) rather than the full training period — deliberately
sized down, per an explicit scope decision, to something that
completes in one interactive session (~200 HRRR pulls) while
meaningfully exercising the pipeline at real scale, not the full
multi-year archive (that's further-out future work).

- **Date selection**: 12 real high-activity dates (one busiest day per
  year 2014–2025, by SPC report count that day — e.g. 2014-09-01 IA
  outbreak, 2021-12-15, 2024-12-29) + 13 genuine quiet dates (verified
  zero SPC reports that day, spread across years and seasons — the
  first time this project has included real negative-class diversity;
  every previous dataset was 100% positive). One run per date (20:00
  UTC, a fixed representative hour for every date — a deliberate
  simplification, not a per-event-tuned choice, same spirit as the
  single-representative-fxx-per-bin approximation above).
- **A real date-selection bug was caught before it silently corrupted
  the dataset**: `2014-02-10` was initially picked as a quiet date
  without cross-checking it against the verified HRRR archive start
  (~Aug 15, 2014) — it's *before* the archive begins, so the pull
  correctly failed loudly (`"Did not find"` from Herbie) rather than
  silently returning wrong data. Fixed by swapping in `2014-11-10`
  (verified quiet AND within the archive) before re-running. Any
  future date selection must check both conditions.
- **Result**: 44 train samples (13 active / 31 quiet — `QuietMapDownsampler`
  finally does real, non-trivial work for the first time) / 6 val
  samples (3 active / 3 quiet), split 44/6 by the deterministic
  mod-20 rule with 0 dropped by the buffer safeguard (the 25 dates are
  naturally well-separated).
  `data/processed/training_dataset_train_scaled.nc` /
  `training_dataset_val_scaled.nc`.
- **Training result** (`models/tornado_cnn_scaled.pt`, 50 epochs,
  `quiet_keep_fraction=0.5`): loss and val loss both converge smoothly
  to ~0.0002–0.0003 — much lower in absolute terms than the 10-sample
  pilot's ~0.0007/0.0017, but this is **not** a sign of better fit; a
  mostly-quiet dataset naturally has lower mean loss regardless of
  discriminative quality (most cells are easy true-negatives).
- **The honest, important finding**: probability separation between
  positive and negative cells on the **held-out val set** is much
  weaker than the pilot's training-set-only numbers ever showed —
  ranging from ~2.5x (one val sample) to ~50x (another) rather than
  the pilot's consistent ~40–90x, and one positive cell in a 14-cell
  val sample got a probability of ~9e-6 (essentially zero — a clear
  miss). This is the first genuine generalization signal this project
  has produced, and it says plainly: **44 training samples is still
  not enough.** Not addressed here — surfacing it accurately is the
  point of having a real val set; don't paper over it with a rosier
  training-loss-only story.
- Scaling further toward the full ~Sept 2014 – Sept 2025 training
  period (see "Time binning" above) is the natural next step, now that
  the archive-availability question is resolved and this smaller
  scale-up has validated the mechanics.

### Full-scale build v1 (superseded — see v2 below)

Built via `scripts/02_dataset_build/build_scaled_dataset.py` + `src/tornado_predictor/build_scaled.py`
(see that module's docstring for the manifest-based resumable design).
Per explicit user direction ("everything all active and quiet days...
happy to have this running in the background"), this covers **every**
UTC calendar date with >=1 SPC tornado report in the ~Sept 2014–Sept
2025 archive window (2,044 active dates), plus 300 randomly-sampled
quiet dates (`seed=0`) — 2,344 dates total, one HRRR run per date
(20:00 UTC, same fixed-representative-hour simplification as the first
scale-up). Ran unattended over ~30 hours wall time.

- **2,279 of 2,344 dates succeeded (97.2%); 65 failed** after
  `features.py`'s internal retries were exhausted. **Failures cluster
  entirely in 2014–2018** (13 in Sept 2014 near the archive-start edge,
  the rest scattered through 2015–2018) — **zero failures from 2019
  onward**. Consistent with the earlier-discovered Sept 2014
  reduced-field-set issue (see "Time binning" above): the earliest
  years of AWS's HRRR archive are less complete/consistent than later
  years, not a bug in this pipeline.
- **Split result**: 3,578 train / 682 val / 298 dropped by the buffer.
  1,660/3,578 train maps (46%) and 325/682 val maps (48%) were
  "active" (>=1 positive cell), per-cell positive rate ~1.5e-04.
- **Superseded by v2 below.** A data-exploration pass
  (`notebooks/data_exploration.ipynb`) surfaced that only 46-48% of
  maps built from dates specifically chosen for having a confirmed
  report were actually "active" — lower than expected. Auditing found
  the cause: this build's one-fixed-20:00-UTC-run-per-date design
  missed a real tornado on **27.2% of active dates entirely** (541 of
  1,989), because most active dates have multiple reports spread
  across many hours (median spread ~10.8h on multi-report days; only
  46% fit inside one 8h window) — the single fixed run's window simply
  didn't bracket most of them. Not a labeling bug (`labels.py` was
  already correct); the bug was in which of a report's 8 valid
  candidate runs this build chose to actually pull.
- Archived, not deleted, for reference:
  `data/processed/training_dataset_{train,val}_full_v1_superseded.nc`.

### Full-scale build v2 (report-covering runs, canonical)

Fixes v1's run-selection gap. `build_scaled.select_active_runs` uses a
greedy interval-covering algorithm per active date: anchor a run at
the earliest not-yet-covered report's hour, skip every report within
the next 8h (already covered), repeat for whatever's left — the
minimum number of runs needed so *every* report that date is captured
by some run's 0-8h window (see `build_scaled.py`'s docstring for the
full derivation). Checked against the real report data before
launching: only 3,209 runs needed for 2,163 active dates (1.48x, not
8x) — 1,305 dates need just 1 run, 670 need 2, 188 need 3. Quiet dates
are unchanged (still exactly 1 fixed-hour run each — no report to
anchor to, so any hour is equally representative). The unit of work
changed from "date" to "run" (`process_one_run`, keyed by full
init_time, since a date can now need multiple runs); `finalize` still
groups by calendar date for the train/val split, so multi-run dates
split as one unit, unchanged from before.

- Validated on real data before the full run: pulled 5 real runs
  across 2 known multi-report dates and confirmed 9 of 10 resulting
  samples came out active (vs. what v1's single-run design would have
  caught).
- **3,335 total runs targeted** (3,035 active + 300 quiet) vs. v1's
  2,344 — 1.4x the volume, as predicted. **3,183 succeeded (95.5%)
  after one retry pass** (`--retry-failed` recovered 40 of the first
  attempt's 192 failures — most were transient `EOFError`/
  `ConnectionError`/`PrematureEndOfFileError`, not archive gaps like
  v1's failures were; 152 still failed, scattered across 2014–2021 and
  2025 with no strong pattern, not investigated further).
- **Split result**: 5,078 train / 980 val / 308 dropped by the buffer
  — meaningfully more samples than v1 (6,058 vs. 4,260 total) from
  fewer targeted active dates, because runs are no longer wasted on
  windows with nothing in them.
  `data/processed/training_dataset_train_full.nc` (3.0GB) /
  `training_dataset_val_full.nc` (596MB) — promoted to the canonical
  filenames (v1's are archived, see above).
- **Class balance improved substantially, as intended**: 3,208/5,078
  train maps (**63.2%**, up from v1's 46%) and 631/980 val maps
  (**64.4%**, up from 48%) are active. Per-cell positive rate ~2.13e-04
  in both splits (train 12,090/56,761,884; val 2,327/10,954,440),
  consistent between splits. Zero NaNs, zero degenerate all-zero
  fields.
- **Not yet done**: retraining `TornadoCNN` on this dataset and
  re-running the generalization evaluation that flagged v1's smaller
  25-date scale-up as insufficient (see "First scale-up" above) — this
  build only produces the dataset; `scripts/03_architecture_search/train_model.py --val-dataset ...`
  against it is the natural next step, not performed automatically
  here.


### Full-scale build v4 (expanded quiet dates + one retry pass)

Same `scripts/02_dataset_build/build_scaled_dataset.py` / manifest (`data/interim/scaled_build_v2/`),
re-run with `--quiet-mode all` (every remaining quiet date, not 300 sampled) and
then `--retry-failed`. Outputs: `data/processed/training_dataset_{train,val}_full_v4.nc`
(the canonical `_full.nc` files are untouched).

- **Pulls**: first pass 4,822 done / 228 failed; after one retry pass
  **4,919 done / 131 failed** (97 recovered). Of the original 228 failures,
  107 were `ValueError: No index file was found` (the `.idx` is absent from
  the AWS archive -- will not recover on retry), 49 `EOFError`, 38
  `PrematureEndOfFileError`, 24 `FileNotFoundError`, 10 `ConnectionError`
  (the last four are transient/cache). 195/228 failures are 2014-2018, same
  early-archive pattern as v1/v2. Remaining 131 not pursued further.
- **Split** (2-way, mod-20 rule, 24h buffer): 7,652 train / 1,478 val /
  708 dropped.
- **Sanity check (verified)**: the 14 core features have zero NaNs, no
  all-constant samples, plausible ranges; train/val means nearly identical.
  Active maps **43.4% train / 44.5% val** (down from ~63% -- expected, quiet
  expansion); per-cell positive rate **1.44e-04** in both (was 2.1e-04).
- **UH columns are mostly NaN in the v4 `_full_v4.nc` files, but NOT because the new runs lack UH (CORRECTED --
  an earlier version of this note had it backwards):** the 3,182 original runs staged in
  `data/interim/scaled_build_v2/` carry only the 14 base features (their UH-augmented copies live in
  `data/interim/scaled_build_v3/`), while the 1,737 newer runs carry all 21 features natively. The v4 train/val
  files were assembled from `scaled_build_v2/`, so the old runs' UH is NaN there (77% for `uh_0_2km`/`uh_0_3km`,
  65% for `uh_2_5km`/`uh_layers_available`; the 0-2/0-3 km layers are additionally missing before 2018-07-13).
  `uh_layers_available` is NaN (not 0/1) for those. **Do not train on the 21-feature `_full_v4.nc` files as-is
  (drop the 7 UH columns, as the CV runs do).** For any UH work use `data/interim/scaled_build_uh_merged/`
  (symlinks: the `_v3` augmented copy for the old runs, the native file for the new ones): all 4,918 of 4,919 runs
  carry UH, `uh_2_5km` has zero NaN, `uh_0_2km` is all-NaN exactly for pre-2018-07-13 runs, and the 14 base
  features and labels are identical to the `_v2` copies (checked on 40 runs); the one exception is 2022-10-14.
- `scripts/05_year_cv_diagnostics/cross_validate_year.py` gained `--drop-columns` (needed because
  staged runs mix with/without UH, and xr.concat silently NaN-fills).
- **v4 year-holdout CV result (12 folds, `models/cv_year_holdout_v4/`, 14
  features, identical recipe): v4 is statistically indistinguishable from the
  original -- not better, not worse.** As reported (different test sets) AUC-PR
  is 0.0442 +/- 0.0142 vs. the original 0.0486 +/- 0.0180 (-9%, v4 wins 3/12
  folds), but those test sets differ. Scoring both model sets on the *same*
  samples (`scripts/05_year_cv_diagnostics/compare_cv_old_vs_v4.py`, inference only; results in
  `models/cv_year_holdout_v4/like_for_like.json`; analysis in
  `notebooks/cv_v4_vs_v2_comparison.ipynb`): AUC-PR **-4.0%** on the full v4
  test years (5/12 wins, paired-t p=0.54) and **-3.7%** on the old-run subset
  (5/12, p=0.55); best-F1 flat (+0.6%/+1.1%), AUC-ROC flat (+0.1%). Per-fold
  swings are +/-25-35% in both directions (v4 +33% in 2016, +36% in 2024; -26%
  in 2020, -23% in 2025), so 12 single-seed folds can't resolve a ~4% mean
  difference. Old models on the old-run subset reproduce the original CV's
  AUC-PR to within 5e-7, validating the like-for-like setup. **Cited number
  for the 14-feature U-Net stays ~0.045-0.049 AUC-PR**; the ~1,700 added quiet
  runs gave no measurable ranking gain (and no measurable harm; dataset is
  closer to natural class balance). Correction to an earlier guess: the lower
  base rate (test positive rate -34%) only explains ~-6% of AUC-PR with the
  model held fixed, not a large mechanical artifact. Calibration unchanged
  (mean positive-cell p 0.1365 vs 0.1364). The calendar-year skill trend
  persists on identical test samples (r=0.79 old, 0.87 v4). Untested:
  `quiet_keep_fraction=0.2` means only ~20% of the new quiet maps are seen per
  epoch. (Per-seed variance was subsequently measured and is large -- see "Seed-variance study and ensembles" below.)
  **Precision/recall at each model's own best-F1 threshold** (now reported
  alongside AUC-PR; already computed by `evaluate.compute_metrics`): at that
  operating point both model sets flag ~1 in 5 positive cells (recall ~0.19-0.20)
  at ~9% precision (~10 flagged cells per hit); best-F1 thresholds ~0.21 for
  both. On the old-run subset precision/recall differ by 1-2% (p>0.7). On the
  full v4 test years v4 has +20% recall (0.162 -> 0.195, 8/12 wins, p=0.061,
  suggestive only) with -7% precision (p=0.25); the old models' recall falls
  when scored on the quieter v4 test mix while v4 models' doesn't -- a
  threshold-dependent operating-point effect with no matching AUC-PR gain. Not
  a matched-recall/precision comparison (each model uses its own threshold,
  tuned on the data it is scored on); a precision-at-fixed-recall comparison
  from the full PR curves is the cleaner follow-up, not done.


