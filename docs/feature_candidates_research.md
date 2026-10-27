# Candidate HRRR features beyond the current 14: literature and availability review

Research notes, written 2026-10-22. Informational only: nothing here changes a
locked methodology decision, and no code or data was touched. The question was
which additional HRRR forecast fields are most worth adding to the dense-grid
U-Net, with the user's hypothesis (HRRR pressure / pressure tendency) treated
as something to test against the literature, not as established.

Method: papers were downloaded and read as text where the publisher allowed it
(NOAA repository PDF copies, SPC-hosted copies, arXiv). HRRR availability was
checked against real `.idx` inventories on the public AWS archive (text files
only, no GRIB): the seven requested dates, plus a scan of one file every 5 days
from 2014-09-02 to 2025-11-28 (819 files) and a daily scan of 2014-09-01 to
2025-12-01 for the fields whose presence changes over time. Reading status for
each source is in the Sources list: **full** = read the body, **partial** = read
the sections cited here only, **abstract** = abstract or search snippet only.

## Bottom line

- **The best-supported additions are the classic convective-parameter
  ingredients we are missing**, not pressure: mixed-layer and most-unstable
  CAPE/CIN, lifting condensation level (LCL), 0-1 km bulk shear, and a
  composite significant tornado parameter (STP) built from them. All are
  available for the whole 2014-09 to 2025 period with no version break in
  search string, cost 1-4 extra pulls each, and have moderate (proximity
  sounding and random-forest) evidence. None has been ablated in a CAM-based
  neural-network tornado study that I could find.
- **Simulated composite reflectivity (`REFC`) is the one candidate aimed at the
  known weak spot (timing within a severe day).** Two CAM-ML papers found
  storm-scale "explicit" fields the most influential inputs. But the same
  literature rates updraft helicity highly and this project's UH test was null,
  so transfer is uncertain; and explicit fields have a documented HRRRv3 to v4
  distribution break.
- **Pressure: the literature does not support it as a high-value feature, and
  does not refute it either.** Pressure *level* as an ML input has mixed
  results (high importance in one random forest, among the least important in
  one HRRR CNN). I found **no study that tested forecast pressure tendency as an
  ML input for tornadoes or severe weather.** Supporting evidence for pressure
  falls is classical and observational (1958 mesolow case study, 12-hour
  surface pressure falls on outbreak days). It is cheap (1 pull per fxx, the
  tendency is a difference of snapshots already being built), so it is worth a
  controlled ablation, but expect no more than a modest effect.
- **Several newer HRRR fields are not usable for the full period**: 0-3 km CAPE,
  effective-layer helicity (`EFHL`), critical angle, ESP appear only from
  2020-12-03 (HRRRv4); level of free convection only from 2021-12-22.
- **Cost warning (my estimate):** the live smoke run in CLAUDE.md pulled 16
  fields in 31 s (~2 s per pull). The current plan is 8 pulls x 8 fxx per run.
  Each added field adds 1 pull x 8 fxx x ~4,900 runs, about 40,000 pulls or
  roughly 1 day of sequential pulling per field. Pick the set deliberately.

## Evidence key

Strength refers to evidence that the field adds skill *beyond CAPE + shear +
SRH* for tornado-relevant prediction: **Moderate** = shown in at least one
read paper with an importance or discrimination result; **Weak** = included in
papers but no field-specific result, or conflicting; **None found** = no
supporting paper located. Everything marked "(inference)" is my reasoning, not a
paper's finding.

## 1. Ranked candidates

Availability strings are exact Herbie/regex `search_this` matches (verified
unique per file in the scan). "Pulls" = extra Herbie calls per fxx. Dates are
the earliest day seen present; "all" = every sampled day from 2014-09-02.

| # | Candidate | Evidence for skill (paper-level) | HRRR availability and breaks | Cost | Priority |
|---|---|---|---|---|---|
| 1 | **Mixed-layer CAPE/CIN** and **most-unstable CAPE/CIN** | Moderate. STP/SCP are built on MLCAPE and MLCIN (Thompson 2003, 2012, full/partial). Gensini 2021: CAPE_ML is rank 6 (RF) and 3 (logistic) for significant-vs-weak tornado classification. Sobash 2020 uses MUCAPE and MLCIN but reports no per-field importance. Loken 2022 derives SB/MU CAPE ratio (elevated convection). | `CAPE:180-0 mb above ground`, `CIN:180-0 mb above ground`, `CAPE:255-0 mb above ground`, `CIN:255-0 mb above ground`, optional `PLPL:255-0 mb above ground`. All present on all dates. v2 (2016-08) and v3 (2018-07) changed physics, not names. | 4 pulls (6 with PLPL and 90-0 mb) | **P1** |
| 2 | **Simulated composite reflectivity** (`REFC`), optionally hourly max `MAXREF` | Moderate but transfer-uncertain. Sha 2024 (full): "storm-scale explicit predictors and CAPE are the most influential" (permutation importance, HRRR CNN, any-severe label). Loken 2022 (full): 1-km max reflectivity is 4th for tornadoes in an RF. Conflicts with this project's null UH result. | `REFC:entire atmosphere`, `MAXREF:1000 m above ground`. All dates. **Distribution break at HRRRv4** (SCN 20-46: implicit-explicit vertical advection, latent-heating limit removed; Sha 2024 reports explicit fields much weaker in v3). At fxx 0, `REFC` is the analysis (HRRR assimilates radar reflectivity); `MAXREF` at fxx 0 is labelled `0-0 day max fcst` (check it is not empty before use). | 1-2 pulls | **P1** (the only timing-oriented candidate) |
| 3 | **Surface-based LCL height** | Moderate. Thompson 2003: moisture and shear within 1 km discriminate nontornadic from significantly tornadic supercells; MLLCL term in STP. Thompson 2012 and Grams 2012: lower MLLCL with significant tornadoes. Included in Sobash 2020 and Flora 2021 inputs (no importance reported). Literature uses *mixed-layer* LCL; HRRR gives surface-based. | `HGT:level of adiabatic condensation from sfc`. All dates. Alternative at zero pulls: compute from `t2m`/`d2m` already pulled (inference: do it per native pixel before 13x13 pooling). | 1 pull (or 0) | **P1** |
| 4 | **0-1 km bulk wind shear** | Moderate. Thompson 2003/2012 (STP/SCP low-level shear and SRH terms); included in Sobash 2020; Gensini 2021 has 0.01-3 km shear in the top 10. HRRR provides 0-6 km shear already; 0-1 km is a different layer. | `VUCSH:0-1000 m above ground`, `VVCSH:0-1000 m above ground`. All dates. (v2 TIN 16-26 fixed a layer-label order; names unchanged.) | 2 pulls | **P1** |
| 5 | **Fixed-layer STP** (derived: MLCAPE/1000 x shear06/20 x SRH01/100 x LCL term, CIN term) | Moderate. Loken 2022 (full): after 0-3 km UH, STP is the first-listed "other important" tornado predictor; index-only RFs beat environment-only RFs, and storm+index RFs were slightly better than all predictors. Thompson 2003/2012 define and validate it. Our own untrained STP-like baseline scores AUC-PR ~0.016 vs. U-Net 0.044 (CLAUDE.md). | Derived from items 1, 3 and fields we have (`HLCY:1000-0 m`, 0-6 km shear). No new pulls beyond 1, 3. Compute on the native 3 km grid, then pool mean/max (inference: a product of pooled fields is not the pooled product). | 0 extra | **P1** (after 1, 3) |
| 6 | **Temporal statistics and tendencies of existing fields** (max/min/mean over the bin's fxx; hourly or bin-length differences of CAPE, SRH, shear, T/Td) | Weak. Sobash 2020 used 0/1/2-hour temporal max/mean windows (no differences) and reported group-level ablation only. Hill 2020 sampled fields at several times. No study found testing differences. Grams 2012 (partial): 850 hPa moistened 2-4 C in the 12 h before significant tornadoes. This project's validation case saw SRH more than double in the hour before a tornado (CLAUDE.md, one case). | Needs the all-fxx pulls already being built. | 0 pulls | **P2** (free; test) |
| 7 | **Surface pressure and its tendency** | Weak/mixed, none for tendency. See section 2. | `PRES:surface` all dates (unique match). MSLP has a name break (below). | 1 pull | **P2** |
| 8 | **10 m winds and derived convergence** | Weak. Hill 2020, Flora 2021 (adds 10 m divergence), Sobash 2020 (max 10 m wind) include them; no per-field tornado importance read. Boundary convergence is a plausible initiation cue (inference). | `UGRD:10 m above ground`, `VGRD:10 m above ground`. All dates. v3 (2018-07-12) changed the 10 m wind and 2 m T diagnostics (SCN 18-58), so these (and `t2m`, LCL from t2m/d2m) have a step. | 2 pulls; divergence needs native-grid derivation | **P2** |
| 9 | **850/500 hPa winds** (LLJ, deep flow) | Mixed. Gensini 2021: 500 and 850 hPa wind speed are rank 1-2 (RF) for significant-vs-weak tornado classification. But Sobash 2020: removing upper-air fields (925-500 hPa winds/T/Td) "had a negligible impact" on skill. | `UGRD:850 mb`, `VGRD:850 mb`, `UGRD:500 mb`, `VGRD:500 mb` (925 mb too). All dates. | 4 pulls | **P3** (largely redundant with shear/SRH) |
| 10 | **Lapse rates** (700-500 hPa from `TMP`/`HGT`) | Mixed/weak. Gensini 2021: 2-6 km lapse rate rank 5 (tornado RF) for significant-vs-weak classification. Loken 2022: 700-500 lapse rate judged relatively unimportant (for hail, subsumed by SHIP). Low-level (0-3 km) lapse rates not directly in HRRR. | `TMP:700 mb`, `TMP:500 mb`, `HGT:700 mb`, `HGT:500 mb` (also 850). All dates. | 4 pulls | **P3** |
| 11 | **Precipitable water** | None found for tornadoes (Hill 2020 includes `PWAT`, tornado importance not stated). | `PWAT:entire atmosphere`. All dates. | 1 pull | **P3** |
| 12 | **Static context** (lat, lon, day of year; elevation) | Moderate that models use it: Sobash 2020, Hill 2020, Sha 2024 include them; Loken 2022: EM random forests "place much more importance" on lat/lon (learned report climatology). **Do not add init hour / forecast hour**: see section 4 (sample-mix hazard). | `HGT:surface` (elevation) all dates; lat/lon/day from the grid. | 0-1 pull | **P2** (lat/lon/doy only) |
| 13 | **HRRRv4-only fields**: 0-3 km CAPE, effective-layer SRH, critical angle, ESP, LFC | Moderate in literature (Gensini 2021: effective-layer SRH rank 10, CAPE_EL03 rank 5 logistic; Thompson 2012 effective-layer STP; Coffer 2019 near-ground SRH: abstract only). | `CAPE:0-3000 m above ground`, `CANGLE:0-500 m above ground`, `ESP:0-3000 m above ground`: 2020-12-03 on. `EFHL:surface`: 2020-12-03 on but absent on ~10% of days through 2022-11-15. `HGT:level of free convection`: 2021-12-22 on (339/365 days in 2022). | 1 pull each | **Skip** for full-period training; only for a 2021-2025 side study |
| 14 | **Low-level rotation and updraft diagnostics**: `RELV`, `MAXUVV`, UH layers | Weak. Sobash 2016 (abstract): 1 km vorticity and 0-3 km UH beat 2-5 km UH as tornado surrogates. Loken 2022: MAXUVV/UH important for wind. This project: UH null across seeds. | `RELV:1000-0 m above ground` from 2018-07-13 only. `MAXUVV` label changes at v3 (`400-1000 mb above ground` to `100-1000 mb above ground`, same field per NCEP "lowest 400 hPa", my inference). | 1 pull each | **P3** (same availability problem as UH) |
| 15 | `LTNG`, `HAIL`, `APCP`, `HPBL`, cloud and soil fields | None found for tornadoes. `LTNG` changes level label at v2; `APCP` matches 2 records before 2016-08-23. | See scan | 1 pull each | **Skip** |

Suggested first bundle (my recommendation): rows 1-5 are 8 pulls (4 CAPE/CIN, 1 LCL,
2 shear, 1 `REFC`) which doubles the current 8 pulls per fxx; add row 7
(`PRES:surface`, +1) if the pressure ablation is wanted. Rows 6 and 12 are free.
Whatever is added, the project's own rules apply: compare against >=3-seed
ensembles, AUC-PR primary, not single-seed deltas (the noise band is ~19-25%).

## 2. Pressure specifically

### What HRRR provides (verified in `.idx` and NCEP/NOAA notices)

| Field | Search string | Available | Notes |
|---|---|---|---|
| Surface pressure | `PRES:surface` | all dates 2014-09-02 on | Terrain-dependent level; **no reduction step**. Best base for a tendency. |
| Sea level pressure, old | `PRMSL:mean sea level` | 2014-09-01 to 2016-08-23 | Replaced at v2. |
| Sea level pressure, new | `MSLMA:mean sea level` | 2016-08-24 on | NCEP table: "MSLP (MAPS System Reduction)". TIN 16-26: HRRR's `PRMSL` "is replaced by an alternate sea level pressure field" (Disc 0, Cat 3, No 198) at v2. A level break; a tendency of either is not safe across it (inference). |
| Pressure of lifted parcel | `PLPL:255-0 mb above ground` | all | Parcel level, not a pressure state. |
| Isobaric heights | `HGT:500 mb`, `HGT:700 mb`, `HGT:850 mb`, `HGT:1000 mb` | all | In the sfc file (full 3-D in prs file). 500 hPa height change is a synoptic-forcing proxy. |
| Pressure tendency | none | never | No 3-hour tendency field exists in HRRR inventories (checked 2014 and 2024 full lists). TIN 16-26 says RAP's 3-h tendency was dropped at RAP v3; HRRR was never listed. Must be differenced. |

Known artifact: **SCN 22-68** (full text): the Dec 2020 HRRR upgrade "inadvertently
included an incorrect specification of the height of the Great Lakes, with a
value of 0 used at all Great Lakes grid points. This has led to problems with the
low-level wind, sea level pressure and other near-surface fields", fixed
2022-07-19. Our grid includes lake cells; 2020-12-02 to 2022-07-19 near-surface
fields (incl. possibly `PRES:surface`, `t2m`, 10 m winds) are suspect there. The
notice names sea level pressure, not surface pressure, so how far it reaches
into `PRES:surface` is unverified.

### What the literature found

- **Hill, Herman, Schumacher 2020** (full): a Day-1 GEFS random forest with MSLP
  as a predictor lists "APCP, MSLP, and SRH" among variables with high feature
  importance for tornadoes; authors suggest MSLP "may be characterizing the
  synoptic environment". Caveats (inference): impurity-based importance, 1 degree
  ensemble-reforecast grid, daily labels, MSLP *level* not tendency.
- **Sha, Sobash, Gagne 2024 (arXiv)** (full): HRRR CNN, MSLP among the inputs;
  permutation importance finds "2-m dewpoint, MSLP, and CIN are the least
  influential predictors", explicit fields and CAPE the most. This is the closest
  analogue to our setting and it is a mild negative for pressure *level*.
- **Sobash, Romine, Schwartz 2020** (full): surface pressure is one of 42 base
  inputs; no per-field importance, only group ablations (see section 3).
- **Coffer et al. 2020** (full, 4-page proceedings): random forest on RUC
  sounding profiles finds the pressure variable has "little impact". Weak
  relevance: these are pressures at 37 heights within a sounding, nearly a
  function of height, not surface pressure change.
- **Grams et al. 2012** (full): cites Miller (1972) that outbreak days show
  "greater midlevel height and surface pressure falls"; Table 2 gives a median
  12-h surface pressure fall of 5 hPa for significant-tornado outbreak days (90th
  percentile 10 hPa; 0 at the 10th). Descriptive only: I found no test of surface
  pressure falls as a discriminator against significant hail/wind events in the
  text. Also: almost half of significant tornado events had small 500 hPa height
  falls (<30 m), so large-scale falls are not necessary.
- **Magor 1958** (partial): case study tracking a mesolow along ~600 miles and
  into a tornado; cites earlier work that "micro-lows" appeared "an hour or two"
  before tornadoes. States that hourly changes in *sea level* pressure "are
  smoothed excessively by the introduction of the correction", favoring
  altimeter-setting changes. Observational, one case, 1950s network.
- **No paper found** that tests HRRR (or any CAM) forecast pressure tendency as a
  tornado predictor, or verifies whether HRRR forecasts mesolows. Searches for
  that gave only adjacent case studies and climatologies.

### Honest reading, and how this relates to the user's observation

The user's radar-detection and weather-station evidence is *observed* local
pressure behavior near an existing storm. The literature agrees that
storm-scale and mesolow pressure falls accompany tornadic storms, but those
signals are tens of kilometers or smaller. HRRR's forecast surface pressure
pooled over a 39 km cell is a model field; whether it carries a usable
mesolow signal several hours ahead of storms is **unverified** and not tested
anywhere I could find. The part of pressure behavior that HRRR can credibly
forecast is synoptic-scale (falls ahead of a lee cyclone, 12-hour scale), which
other inputs (shear, SRH, CAPE, and the U-Net's own spatial context) likely
already encode (inference).

### What a tendency feature would be built from (inference, for a test)

- Base: `PRES:surface` at each fxx in a bin (bin 0: fxx 0-3; bin 1: fxx 4-7),
  pooled to the 39 km cell (mean, plus a within-cell std or min as a mesolow
  proxy). Use surface pressure, not MSLP: no reduction, no v2 name break.
- Candidates: `P(fxx last) - P(fxx first)` within the bin (3 h change), the
  bin-to-bin change, and the minimum hourly change; optionally a 500 hPa height
  change from `HGT:500 mb`.
- Risks to handle: (1) the diurnal/semidiurnal pressure tide is a domain-wide
  signal of order a hPa, comparable to the ~0.4 hPa/h implied by Grams'
  5 hPa/12 h, so subtract a domain or latitude-band mean tendency; (2) fxx 0 to 1
  differences may contain adjustment noise (unverified); (3) the Great Lakes bug
  period; (4) the sample-mix issue below can let any time-correlated feature act
  as a shortcut.
- Test design: add to the 14 base features, >=3-seed ensemble vs. ensemble,
  AUC-PR primary, 12-fold year CV, rule written before looking. Predicted size
  of effect: small. If it is below the ~12% minimum detectable effect for a
  3-seed paired comparison (CLAUDE.md), call it "no detectable effect".

**Verdict:** not supported as a priority; not ruled out. Treat as a cheap
ablation (P2), not as the expected best addition.

## 3. What the papers used and what they found

| Paper | Inputs relevant here | Label | Importance / ablation result |
|---|---|---|---|
| Sobash et al. 2020 (full) | 42 base predictors on an 80 km grid: forecast hour, day of year, lat, lon, SBCAPE, MUCAPE, SBCIN, MLCIN, 0-6 km and 0-1 km shear, MLLCL, 0-1 and 0-3 km SRH, 2 m T/Td, surface pressure, MUCAPE x shear, STP, 700-500 hPa lapse rate, hourly max 2-5/0-3/0-1 km UH, max up/downdraft below 400 hPa, max 10 m wind, hourly precip, 925/850/700/500 hPa u, v, T, Td. Plus 132 neighborhood predictors: means/max within +-1 or 2 grid boxes and 0/1/2 h windows (174 total). | Any severe report within 40 or 120 km and 2 h; trained leave-one-year-out | Group ablations only: removing upper-air fields negligible; removing neighborhood or explicit fields larger; UH-only NN still beat the UH smoothing baseline; environmental inputs matter more during spin-up, explicit ones overnight. No per-field ranking. |
| Sha et al. 2024 (full, preprint) | HRRR v3/v4, 64x64 patches per 80 km box: CREF, 0-2 and 2-5 km UH, APCP, 10 m wind, column graupel, MSLP, 2 m T/Td, SBCAPE, SBCIN, 0-1 and 0-3 km SRH, 0-6 km u/v shear, lat/lon/elevation | Any report in an 80 km cell, 4 h window (report mapped to t-2..t+1) | Permutation importance: explicit fields and CAPE most, 2 m Td, MSLP, CIN least. Notes HRRRv3 UH magnitudes much lower than v4. |
| Hill et al. 2020 (full) | GEFS reforecast: CAPE, CIN, MSLP, PWAT, 2 m T/q, 10 m winds, 0-850/500 hPa shear, APCP, plus Day 1 RH, LCL height, 0-850 hPa SRH; neighborhood up to 3 boxes; lat/lon/julian day | Report within 40 km, daily | CAPE most predictive overall; for tornadoes shear (SHR850), APCP, MSLP, SRH rank high; SRH top in the East. |
| Loken et al. 2022 (full) | HREFv2.1: 32 fields incl. reflectivity, echo top, UH, MUCAPE/MUCIN, SRH, lapse rate, STP, SCP, SHIP, EHI, critical angle proxy, lat/lon | Report-based, next day | Tornado: 0-3 km UH top, then STP, 2-5 km UH, 1 km reflectivity, 0-1 km SRH. Environment-only RFs worst; storm+index slightly better than all fields. |
| Flora et al. 2020 preprint (partial) | WoFS 0-3 h: UH, composite reflectivity, 10 m divergence, MLCAPE/MLCIN/MLLCL, lapse rates, 850/700/500 hPa T/Td/height | Storm track overlaps a report | Beat a UH-only calibration; I did not extract per-field importances. |
| Gensini et al. 2021 (partial) | 84 ERA5 proximity-sounding variables (not forecasts) | Significant vs. weak tornado | Top RF tornado: 500 hPa wind, 850 hPa wind, 0-500 m SRH; then effective-layer CAPE, 2-6 km lapse rate, ML CAPE, 0-6 km shear. A conditional task, unlike ours. |
| Lagerquist et al. 2020 (partial) | Radar images + RAP proximity sounding (u, v, RH, q, virtual potential temperature), next-hour | Storm-based | No pressure or per-variable result found in the text. |

Synthesis (inference): every CAM or reforecast ML paper uses CAPE, shear, SRH and
2 m T/Td, and most add MUCAPE/MLCIN/LCL/STP and storm-scale fields. None isolates
the marginal value of LCL or 0-1 km shear in a CNN, so the "moderate" ratings
above come from sounding studies and random forests, not from the architecture
we use.

## 4. What the literature does that we do not

- **Inputs:** neighborhood and temporal-window statistics (132 of Sobash's 174
  inputs; the U-Net's receptive field covers space but not within-bin time
  statistics); lat/lon/day-of-year; MU and ML parcels, ML LCL, 0-1 km shear,
  STP/SCP-type indices; low-level rotation and updraft surrogates; reflectivity.
- **Label definitions:** all four ML papers predict *any severe* (tornado, hail,
  wind) or hazard-specific labels with a **spatial tolerance (40 or 120 km) and a
  +-2 h temporal tolerance**; Sha maps each report to four hours. Ours is
  tornado-only, track corridor touching a 39 km cell, no tolerance. The
  neighborhood-dilated verification already done here (k=1, 2) is the
  verification analog, but training labels were left unchanged (changing them
  needs discussion per CLAUDE.md). Sobash 2020 itself notes report biases
  (wind-damage over-reporting, population effects) that tornado-only labels
  mostly avoid.
- **Verification:** Brier skill score against climatology and reliability
  diagrams; ROC area; a surrogate-severe (smoothed UH) baseline; comparison with
  SPC outlooks (Hill 2020, Loken 2020/2022); stratification by hour, region and
  regime (e.g. high-shear low-CAPE); bootstrap intervals. We have AUC-PR, year CV,
  ensembles and untrained baselines, but no smoothed-UH or SPC-outlook baseline
  and no reliability-based calibration beyond the tier study.
- **Already matching them:** leave-one-year-out training (Sobash 2020), a 4 h
  window, report-based labels from SPC, importance by ablation or permutation.
- **Sample-mix hazard for time features (inference from CLAUDE.md):** 70% of quiet
  samples are init 20Z versus 11% of active ones. Adding init hour, forecast hour
  or local time as an input (as the literature does) would let the model learn
  the run-selection design rather than weather. Do not add them unless the
  run-selection design is changed. Day of year and lat/lon are less exposed.

## 5. HRRR availability: how it was verified

Per-date presence in the `wrfsfc` `.idx` (t06z, f01; t00z and other cycles agree,
and `t20z f07` was checked on all seven dates: `PRES`, sea level pressure, `REFC`,
`CAPE:180-0` present every time; `EFHL` only on the last two). Y = present.

| Search string | 2014-09-02 | 2016-06-01 | 2018-06-01 | 2018-08-01 | 2020-11-01 | 2021-01-15 | 2024-05-01 |
|---|---|---|---|---|---|---|---|
| `PRES:surface` | Y | Y | Y | Y | Y | Y | Y |
| `PRMSL:mean sea level` | Y | Y | - | - | - | - | - |
| `MSLMA:mean sea level` | - | - | Y | Y | Y | Y | Y |
| `REFC:entire atmosphere` | Y | Y | Y | Y | Y | Y | Y |
| `CAPE/CIN:180-0 mb above ground` | Y | Y | Y | Y | Y | Y | Y |
| `CAPE/CIN:255-0 mb above ground` | Y | Y | Y | Y | Y | Y | Y |
| `HGT:level of adiabatic condensation from sfc` | Y | Y | Y | Y | Y | Y | Y |
| `VUCSH/VVCSH:0-1000 m above ground` | Y | Y | Y | Y | Y | Y | Y |
| `UGRD/VGRD:850 mb`, `500 mb` | Y | Y | Y | Y | Y | Y | Y |
| `UGRD/VGRD:10 m above ground` | Y | Y | Y | Y | Y | Y | Y |
| `PWAT:entire atmosphere` | Y | Y | Y | Y | Y | Y | Y |
| `MAXUVV:400-1000 mb above ground` | Y | Y | Y | - | - | - | - |
| `MAXUVV:100-1000 mb above ground` | - | - | - | Y | Y | Y | Y |
| `MXUPHL:2000-0` and `3000-0 m above ground` | - | - | - | Y | Y | Y | Y |
| `RELV:1000-0 m above ground` | - | - | - | Y | Y | Y | Y |
| `LTNG:surface` / `LTNG:entire atmosphere` | surf | surf | entire | entire | entire | entire | entire |
| `CAPE:0-3000 m above ground` | - | - | - | - | - | Y | Y |
| `EFHL:surface`, `CANGLE:0-500 m above ground`, `ESP:0-3000 m above ground` | - | - | - | - | - | Y | Y |
| `HGT:level of free convection` | - | - | - | - | - | - | Y |

Record counts per file: 102 (2014, 2016), 132 (2018-06), 148 (2018-08, 2020-11),
170 (2021, 2024). Sampling of 819 days (every 5th day) found every "all dates"
string present exactly once on every sampled day with a file, with these
exceptions: `APCP:surface:0-1 hour acc` matches 2 records through 2016-08-22;
`RELV` absent through 2018-07-08, present from 2018-07-13. Daily scan results
(4,110 days, first of 06/12/00Z with a file): first day with `MSLMA` 2016-08-24
(last `PRMSL` 2016-08-23); `MXUPHL:3000-0`, `RELV`, `MAXUVV:100-1000` first
2018-07-13; `EFHL`, `CANGLE`, `ESP`, `CAPE:0-3000`, `LAYTH` first 2020-12-03
(`EFHL`/`CANGLE`/`ESP` missing on 81 of 1,825 post-v4 days: 33 in 2021, 46 in
2022, 2 in 2024; `CAPE:0-3000` complete); `HGT:level of free convection` first 2021-12-22. 16 days
had no file at any of 00/06/12Z (e.g. 2014-12-13, 2016-08-09/10, 2017-06-28).

### Version breaks (from the NOAA notices, full text)

| Date (first run) | Version | Documented change relevant here |
|---|---|---|
| 2016-08-23 | HRRRv2 (TIN 16-26) | HRRRv1 runs were independent; v2 cycles land-surface states and adds hybrid ensemble DA, radial wind and mesonet assimilation, improved surface-observation assimilation; Thompson, MYNN, radiation updates; sea level pressure field replaced (`PRMSL` to `MSLMA`); output written straight to GRIB2 (encoding differences possible); forecasts 15 to 18 h. |
| 2018-07-12 | HRRRv3 (SCN 18-58) | Hybrid vertical coordinate, improved terrain, new 2 m T, 10 m wind, gust and visibility diagnostics; radar radial velocity and lightning assimilated; forecasts to 36 h on 4 cycles. New output starts 2018-07-13 (UH 0-2/0-3 km, RELV, MAXUVV layer label). |
| 2020-12-02 | HRRRv4 (SCN 20-46) | HRRRDAS 36-member ensemble DA with reflectivity assimilation, implicit-explicit vertical advection permitting stronger updrafts and removal of a latent-heating limit, MYNN update, new lake model; new outputs 0-3 km CAPE, effective SRH, critical angle, ESP, LFC (SCN lists LFC for v4; the archive shows it only from 2021-12-22), DZDT 700, HAIL. |
| 2022-07-19 | Fix (SCN 22-68) | Great Lakes elevation set to 0 m by the v4 change, corrected; near-surface fields affected 2020-12-02 to 2022-07-19. |

Existing features (`t2m`, `d2m`, CAPE, SRH) also cross these breaks; the
calendar-year skill trend in the CV may partly reflect them (not tested).

## 6. Open questions and uncertainties

1. **Does reflectivity transfer here?** Positive evidence is for any-severe
   labels at 80 km; our UH null suggests storm-scale fields may not help a
   tornado-only label. Also needs a per-version normalization decision.
2. **Value of the convective-parameter bundle for a CNN** is unmeasured. Trees
   gain from engineered indices (Loken 2022); a U-Net can learn products but
   may not, especially from 13x13-pooled mean/max. Pooled-then-multiplied
   versus multiplied-then-pooled STP is an open implementation choice.
3. **Pressure tendency signal-to-noise** at 39 km, hourly, 0-8 h lead is
   unknown; whether HRRR forecasts mesolows ahead of storms is unverified; tide
   removal and fxx 0-1 adjustment noise are untested.
4. **Surface-based vs. mixed-layer LCL:** HRRR gives only surface-based LCL, which
   is unrepresentative in stable nocturnal layers; the literature's STP uses
   mixed-layer. MLLCL would need derivation from layer-mean T/Td (not in the
   sfc file).
5. **EFHL, ESP and CANGLE meanings** come from SCN 20-46's wording ("Effective
   Storm-Relative Helicity at the surface", "Enhanced Stretching Potential",
   "Critical Angle"); mapping SCN parameter numbers to the `.idx` names is my
   inference (NCEP's rendered table leaves those rows blank). Their ~10%
   missing-day pattern in 2021-22 is unexplained.
6. **`MAXUVV` label change** (`400-1000` to `100-1000 mb above ground`) is assumed
   to be the same quantity ("lowest 400 hPa"); unverified.
7. **Not reviewed or only abstract-level:** Sobash 2016 (tornado, abstract
   only), Coffer 2019 near-ground SRH (abstract only), Gagne 2019 and Burke 2020
   (hail), Jergensen 2020 (storm mode from radar), Roebber, Clark, Fujita's mesolow
   work, Markowski mobile-mesonet studies, and the published version of Sha 2024.
   AMS journal pages returned 403; some papers were obtained through NOAA/SPC
   copies.
8. **Importance methods differ** (impurity, permutation, tree-interpreter,
   conditional permutation) and are not comparable across papers; none is a
   controlled add-one ablation like the ones this project runs.
9. **Cost:** pull counts and time per pull are from CLAUDE.md's smoke run, not
   re-measured here.

## Sources

Read as text (full unless noted); details in sections above.

- Sobash, Romine, Schwartz 2020, Weather and Forecasting 35, 1981-2000,
  doi:10.1175/WAF-D-20-0036.1 (NOAA repository copy). Full.
- Sha, Sobash, Gagne, arXiv:2310.06045v2 (Mar 2024), https://arxiv.org/abs/2310.06045. Full; preprint.
- Hill, Herman, Schumacher 2020, Mon. Wea. Rev. 148, 2135-2161,
  doi:10.1175/MWR-D-19-0344.1. Full.
- Loken, Clark, McGovern 2022, Wea. Forecasting 37, "Comparing and Interpreting
  Differently Designed Random Forests for Next-Day Severe Weather Hazard
  Prediction" (NOAA repository copy). Full.
- Flora et al., arXiv:2012.00679v1 (Nov 2020), preprint of the 2021 Mon. Wea. Rev.
  paper. Partial (abstract, predictor table).
- Gensini, Converse, Ashley, Taszarek 2021, Wea. Forecasting 36, 2143-2160,
  doi:10.1175/WAF-D-21-0056.1 (NSF PAR copy). Partial (abstract, importance).
- Lagerquist et al. 2020, Mon. Wea. Rev. 148, 2837-2861, doi:10.1175/MWR-D-19-0372.1. Partial.
- Thompson, Edwards, Hart, Elmore, Markowski 2003, Wea. Forecasting 18, close-proximity
  RUC soundings (SPC copy). Partial (abstract, STP definition).
- Thompson, Smith, Grams, Dean, Broyles 2012, Wea. Forecasting 27, convective modes
  Part II (SPC copy). Partial (abstract, STP/SCP, MLLCL).
- Grams et al. 2012, Wea. Forecasting 27, 106-123, significant tornado climatology
  (SPC copy, https://origin-east-www-spc.woc.noaa.gov/publications/grams/sigt-env.pdf). Partial.
- Coffer, Kubacki, Wen, Zhang, Barajas, Gobbert 2020, Proc. Appl. Math. Mech.,
  doi:10.1002/pamm.202000112. Full.
- Magor 1958, Mon. Wea. Rev. 86(3), "A meso-low associated with a severe storm"
  (SPC copy, https://origin-east-www-spc.woc.noaa.gov/publications/magor/mesolow.pdf). Partial.
- Abstract/snippet only: Sobash et al. 2016, Wea. Forecasting 31, 1591-1614
  (low-level rotation, next-day tornado); Coffer, Parker, Thompson, Smith, Jewell
  2019 (near-ground SRH).

NOAA/NCEP documentation (full text read):

- NWS Service Change Notice 18-58 (HRRRv3), https://www.weather.gov/media/notification/pdfs/scn18-58rap_hrrr.pdf
- NWS Service Change Notice 20-46 (HRRRv4), https://www.weather.gov/media/notification/pdf2/scn20-46rap_v5_hrrr_v4_aab.pdf
- NWS Technical Implementation Notice 16-26 (HRRRv2), https://www.weather.gov/media/notification/tins/tin16-26rap_hrrr.pdf
- NWS Service Change Notice 22-68 (Great Lakes elevation), https://www.weather.gov/media/notification/pdf2/scn22-68_hrrr_great_lakes.pdf
- NCEP HRRR wrfsfc inventory (current), https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml
- NOAA GSL HRRRv4 GRIB2 table, https://rapidrefresh.noaa.gov/hrrr/HRRRv4_GRIB2_WRFTWO.txt
- Real `.idx` inventories, `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFF.grib2.idx`
  (and `wrfprsf01` for 2014-09-02 to 2024-05-01), fetched 2026-10-22.
