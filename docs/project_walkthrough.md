# Project Walkthrough: How This Got Built, and Why

A guided tour of this project's full arc — every major piece, in the order
it was built, with the reasoning behind each decision and the real bugs/
findings along the way. If `CLAUDE.md` is the reference manual, this is
the story that explains why the manual says what it says.

## 1. What this project actually is

**Goal**: predict whether a tornado will touch down in a given ~40km
grid cell, 0-8 hours in the future, using HRRR (a numerical weather
model) forecast fields as input.

**Not to be confused with** the sibling `tornet` project, which
*detects* tornadoes happening right now from live radar. Different
problem: forecasting ahead of time (this project, using coarse
atmospheric fields) vs. confirming something already visible (radar
signatures).

**Methodology**: "dense-grid classification," following a real paper
(Sobash et al.). The key design choice, locked in from day one: every
`(grid cell, time window)` pair is one training sample — exhaustively,
across the whole domain — not just cells near known storms. This is
different from `tornet`'s approach (crops built around storm objects).
It means the model has to learn to say "no" correctly across millions
of boring, storm-free cells, which is where most of this project's
hard problems come from (see section 4).

## 2. The spatial grid

HRRR's native grid is ~3km resolution over the continental US (and
parts of Canada/Mexico/ocean at the edges). That's too fine-grained and
too large to train on directly, so it gets coarsened to ~40km cells.

**The decision that mattered**: coarsen by *exact pixel-block averaging*
(13×13 native pixels → one 39km cell), not by reprojecting/interpolating
onto an independently-designed grid. This keeps pooling mathematically
exact — a cell's value really is the mean/max of the native pixels it
contains, zero interpolation error. The tradeoff: the grid's shape is
whatever 13×13 blocks happen to tile the native domain (81×138 cells,
not a clean number), and it extends over ocean/Mexico/Canada at the
edges rather than being clipped to a US land boundary. That "weird"
NW-corner-reaches-134°W behavior looked like a bug the first time it
showed up — it isn't; it's genuine map-projection curvature, verified
by computing the domain's actual corners directly rather than assuming.

## 3. Time binning

Each HRRR run forecasts 0-8 hours ahead, split into two 4-hour windows
(bin 0: hours 0-4, bin 1: hours 4-8).

**The non-obvious decision**: *every single hourly HRRR run* gets its
own pair of bins — not just the 4 standard "synoptic" runs a day
(00/06/12/18Z). This was deliberate, to maximize how many training
samples exist, and it has a real consequence: one tornado report can
become a positive label in up to 8 different runs' sample sets
(whichever runs' 0-8h windows happen to contain it). That's not
double-counting or a bug — it's the intended per-run sampling design,
and it's proven mathematically (`candidate_run_inits` always returns
exactly 8 runs for any report).

## 4. Labels: turning SPC reports into 1s and 0s

Ground truth comes from SPC (Storm Prediction Center) tornado reports —
touchdown point, track length, damage-path width.

**The decision**: credit a report to *every* cell its damage path
crosses, not just the touchdown cell. A tornado's track gets buffered
(by half its damage width) into a corridor polygon, then intersected
against each candidate cell's exact projected boundary. A long-track
tornado (the Dec 2021 Quad-State/Mayfield EF4, ~165 miles) ends up
touching 13 cells this way, not 1.

**The permanent simplification**: SPC has no per-point track timing —
just one touchdown timestamp for the whole track. So spatial matching
uses the full track, but temporal matching collapses to that one
instant. There's no way to fix this without inventing a fake
track-speed model, so it's documented as a real, permanent limitation
rather than something to silently paper over.

**Storage**: only positive labels are stored (a sparse table). Anything
not listed is an implicit 0 — the full 0-space (every cell × every run
× every bin) is billions of rows and was never worth materializing
directly.

## 5. Features: what the model actually sees

HRRR forecast fields, pulled via a library called Herbie, pooled onto
the coarse grid (mean *and* max per field — max catches a sharp local
spike that mean would dilute).

**Original 7 fields** (14 channels after mean/max): CAPE, CIN, 0-1km
and 0-3km storm-relative helicity, 0-6km bulk shear, 2m temperature,
2m dewpoint. Every search string was verified against a *live* HRRR
index file before being trusted — a loose `"CAPE:"` search, for
example, would also match HRRR's other CAPE variants, so exactness
mattered.

**A real scope tradeoff, not an oversight**: each 4-hour bin is
represented by just its *first* hour's snapshot, not an aggregate over
all 4 hours. Pulling and averaging all 4 would be 4x the network cost
for the whole dataset — deliberately deferred.

**In progress right now**: adding updraft helicity (UH) — literature
review found this is the single most-cited direct tornado/supercell
surrogate in this kind of forecast data, and this project had never
used it. Two real discoveries came out of adding it:

- It's genuinely available on HRRR (`MXUPHL` fields), confirmed live.
- But **only from 2018-07-13 onward** for two of the three UH layers
  (0-2km, 0-3km) — binary-searched the exact date, and it lines up
  exactly with NOAA's documented HRRRv3 model upgrade. Only the 2-5km
  layer covers this project's full 2014-2025 archive.

Rather than drop those two fields or shrink the dataset's date range,
the decision was to keep them and represent the gap honestly: they're
stored as real `NaN` for pre-2018-07-13 samples, with an always-present
`uh_layers_available` indicator feature so the model can tell the
difference between "this value is genuinely near zero" and "we don't
know." The NaN gets imputed to 0.0 only at the very last step (inside
`DenseGridDataset`, right before the numbers hit the model) — the
*stored* dataset stays honest about what's real vs. missing.

This is being built without re-pulling the whole dataset from scratch:
a new module only pulls the *new* fields for dates already
successfully processed, cutting new network calls to roughly 30% of a
full rebuild. This job is the one still running in the background.

## 6. Scaling the dataset up — and the bug that mattered most

The path here: a 6-sample pilot → a 50-sample stratified scale-up → a
full build (v1) → a *corrected* full build (v2), which is what
everything currently trains on.

**v1's real bug, found by actually looking at the data, not assumed
away**: v1 pulled exactly one HRRR run per active date, fixed at 20:00
UTC. Auditing found this missed a real tornado on **27.2% of active
dates entirely** — most days with a tornado have multiple reports
spread across many hours, and a single fixed 8-hour window often just
doesn't cover when it happened.

**v2's fix**: a greedy algorithm that covers every report on a date
with the minimum number of runs needed (most dates still need just 1
run; busy days get 2-3). This is why the current full dataset exists as
"v2," not v1 — and why the dataset build script now operates on
"runs" as the unit of work, not "dates."

**Also discovered along the way**: HRRR's AWS archive doesn't actually
start where the raw file listing suggests — the earliest ~2 weeks
(mid-August 2014) exist but are missing CAPE, CIN, helicity, and shear
entirely (only a reduced ~57-field set). The real usable start date,
confirmed by checking exactly which fields exist on which days, is
**September 1, 2014**.

## 7. Handling extreme class imbalance

Positive cells are rare — about 1 in 5,000 cells at this dataset's
current scale, even though roughly half of the *maps* (grid × time
window) contain at least one positive cell. That distinction shaped the
whole strategy:

- **FocalLoss**, not plain cross-entropy — down-weights "easy" examples
  (the millions of obviously-quiet cells) so the loss focuses on the
  genuinely hard/rare positive cases.
- **Map-level downsampling**, not per-cell — you can't drop arbitrary
  pixels out of a spatial CNN's input without breaking its
  convolutional receptive field, so instead: keep every map with at
  least one positive cell, and only train on a fraction of the
  all-negative maps (re-chosen each epoch).

**A real direction-of-reasoning bug**: an early note said the fix for
under-confident positive predictions would be a *lower* `alpha` in
FocalLoss. That turned out to be backwards — re-reading the actual
formula shows `alpha` directly weights the positive class, so a
*higher* alpha should increase positive-class confidence, not
decrease it. A dedicated sweep (5 training runs, only `alpha` varied)
confirmed this empirically. But then a *second* correction: judging
"best alpha" by simple mean-separation metrics (Cohen's d) said alpha
0.9 was the winner — until a broader, more rigorous metric pass
(AUC-PR, which integrates across the whole precision-recall tradeoff
rather than just comparing averages) found the *opposite*: the
original `alpha=0.25` was actually best, and 0.9 was the *worst* of
the five tested. **The project's real, current default is still
alpha=0.25** — both "corrections" are documented, including the one
that reversed the other, because the disagreement itself is a useful
lesson (see section 9).

## 8. Models tried, in order, and what each one taught

| model | what it is | key result |
|---|---|---|
| **Baseline CNN** | 3 plain conv layers, 32 channels, no pooling | First working end-to-end model. Deliberately tiny "receptive field" (~270km) — the model can only see a small neighborhood around each cell. |
| **Width/depth ablation** | Same architecture, tried wider (more channels) and deeper (more layers) | Widening channels alone *didn't help* (scored below baseline). Adding depth *did* help — more receptive field, not more raw capacity, was what mattered. |
| **Dilated convolutions** | Same layer count, but each layer skips pixels at increasing spacing, so receptive field grows much faster per layer | New best result at the time — bigger receptive field (up to ~1,200km) for *fewer* parameters than the deeper plain-CNN version, and no overfitting. |
| **U-Net** | Downsample → compress → upsample, with skip connections, the same shape used in real published severe-weather ML papers | **Current best model.** Beat everything else on every metric, despite having ~14x more parameters than the dilated CNN — its downsample/upsample structure seems to have real value beyond just "more capacity." |
| **Gradient-boosted trees** | A completely different model family — no CNN at all, just per-grid-cell tabular rows | **Lost decisively** to the U-Net (test AUC-PR 0.029 vs. 0.084). Best explanation: a per-cell tabular model has *zero* receptive field by construction, and every CNN experiment above found receptive field is what actually drives results here. |

A consistent thread runs through all five: **receptive field (how much
surrounding context a model can see) mattered every time it was
tested; raw parameter count, on its own, mostly didn't** — and one
config that combined a lot of extra capacity *with* extra depth
actually overfit (the only model in this whole project to show a real
train/val loss divergence).

## 9. How results actually get measured — and why that took real work to get right

Three separate lessons, discovered in this order:

1. **Accuracy is meaningless here.** Every model scores >99.8% accuracy
   — including, implicitly, a model that just always predicts "no
   tornado." At this level of class imbalance, accuracy tells you
   nothing.
2. **AUC-ROC is misleadingly optimistic.** Every model scored
   0.92-0.99 on it, barely distinguishing a good model from a mediocre
   one — because ROC gets computed against the huge, easy population
   of true negatives. **AUC-PR (average precision) is the metric that
   actually separates good from bad here** — it ranges from 0.019 to
   0.084 across every model tried, a much more honest and more
   spread-out picture.
3. **Every reported number, until recently, came from a validation set
   that had been checked repeatedly** across 5 rounds of architecture
   decisions — exactly the kind of number that can quietly become
   over-optimistic. So a genuine three-way split was built: train
   (70%) / test (15%, **carved from dates never touched by any past
   decision**) / val (15%, same dates as always, so old numbers stay
   meaningful). The real test number for the current best model
   (U-Net) is **0.0836 AUC-PR** — notably *higher* than its old val
   number (0.0398), which is a genuine, interesting finding in its own
   right: at this project's current data scale, which specific storms
   land in which split carries real variance. That's direct evidence
   that a future round of cross-validation (averaging over several
   different splits) would give a more trustworthy number than any
   single split — including this new one.

## 10. Where things stand right now

- **Done**: everything in sections 2-9 above.
- **Best model**: the U-Net, `models/unet/tornado_unet.pt` — test
  AUC-PR 0.0836.
- **Running in the background**: adding updraft helicity features to
  the full dataset (section 5) — an incremental augmentation job over
  ~3,183 already-built runs, expected to finish without needing any
  further input. Once done: promote the enriched dataset, retrain the
  U-Net on it, and see whether UH actually moves the number.
- **Not yet done, flagged for later**: hyperparameter tuning per
  architecture (every model so far shared one fixed recipe), sweeping
  the U-Net's own width/depth, neighborhood-aware features for the
  GBT family, and year-based cross-validation to get a more trustworthy
  performance estimate than any single train/test split.

## 11. The meta-lesson, if there's one

Almost every real correction in this project's history came from
*checking a claim against live data* instead of trusting a plausible-
sounding assumption: the grid's "wrong-looking" NW corner, the HRRR
archive's true start date, the alpha-direction mix-up (twice, in two
different directions), the UH archive gap, and the run-selection bug
that quietly dropped 27% of active dates in the first full build. None
of these were caught by being extra careful in the abstract — they
were caught by pulling a real file, running a real number, or
re-deriving a formula by hand and comparing it to what the code
actually does.
