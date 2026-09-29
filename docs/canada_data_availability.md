# Canada tornado data availability (exploratory notes)

Informational only — not a project decision. This project remains
scoped to the US (HRRR + SPC), following Sobash et al. (2020); nothing
here changes that. These are notes from a one-off "is this even
possible" curiosity check, kept in case the project (or a spinoff)
ever wants to explore a Canadian analog.

## Labels: Northern Tornadoes Project (NTP)

[Northern Tornadoes Project](https://www.uwo.ca/ntp/index.html)
(Western University) is Canada's closest equivalent to SPC storm
reports. It exists specifically because Canada's official historical
tornado record undercounts actual occurrence — most of the country is
sparsely populated, so tornadoes over forest/prairie often went
unwitnessed. NTP actively fills that gap using satellite imagery,
crowdsourcing, and drone/aerial damage surveys.

- Database goes back to **1980**; a recent revision added 250+
  previously-undocumented tornadoes.
- [Open Data site](https://ntpopendata-westernu.opendata.arcgis.com/)
  offers KML downloads, no account needed. A more complete "Advanced
  Dashboard" needs an account (`ntp@uwo.ca`).
- Non-commercial use is explicitly permitted and encouraged;
  commercial use requires contacting NTP directly.
- Much smaller/younger than SPC's record, and Canada's overall
  tornado count is far lower — the eventual positive-label count would
  be sparser than what this project already works with.

## Features: HRDPS

Environment and Climate Change Canada runs
[HRDPS](https://eccc-msc.github.io/open-data/msc-data/nwp_hrdps/readme_hrdps-datamart_en/)
(High Resolution Deterministic Prediction System) — their
convection-allowing regional model, ~2.5km resolution (comparable to
HRRR's 3km), covering most of Canada plus a bit of the northern US.
GRIB2, publicly available via MSC Datamart / MSC Open Data.

- **Herbie already has documented support for HRDPS**
  ([gallery page](https://herbie.readthedocs.io/en/2026.3.0/gallery/eccc_models/hrdps.html)) —
  a real practical head start since this project's whole
  feature-extraction pipeline (`features.py`) is built on Herbie.
- Runs up to 4x/day, not hourly like HRRR — no per-run hourly-binning
  design question to make for Canada; HRDPS simply doesn't update that
  often.

## What a Canada adaptation would actually require

Not a config change — parallel modules, with the same
verify-against-live-data rigor this project has already had to apply
to HRRR twice (grid params, and the Sept 2014 field-availability
discontinuity):

- **Different projection.** HRDPS uses a rotated lat/lon grid, not
  HRRR's Lambert Conformal Conic — `grid.py`'s hardcoded LCC constants
  don't transfer. A new grid module would need its own
  live-verified parameters.
- **Grid/spec changed in Nov 2023.** ECCC changed HRDPS's format that
  year, and the grid moved to rotated lat/lon in early 2023. Directly
  analogous to this project's own Sept 2014 CAPE/CIN/HLCY/shear
  discovery (`build_scaled.py`'s `ARCHIVE_START`) — the historical
  archive is likely not self-consistent and would need the same kind
  of direct `.idx`/inventory verification before trusting any date
  range.
- **Field search strings would need re-deriving** from HRDPS's own
  GRIB inventory, the same way `features.py`'s `FIELD_SPECS` were
  derived and locked in against a live HRRR `.idx` file.
- **Licensing is more restrictive than SPC's fully open bulk CSV** —
  NTP's non-commercial-only terms, with a contact-gated path for
  fuller access.

## Sources

- <https://www.uwo.ca/ntp/faqs/where_can_i_find_canadian_tornado_data_and_information.html>
- <https://www.uwo.ca/ntp/index.html>
- <https://ntpopendata-westernu.opendata.arcgis.com/>
- <https://www.cbc.ca/news/canada/london/tornadoes-canada-online-portal-1.7201054>
- <https://herbie.readthedocs.io/en/2026.3.0/gallery/eccc_models/hrdps.html>
- <https://eccc-msc.github.io/open-data/msc-data/nwp_hrdps/readme_hrdps-datamart_en/>
- <https://open.canada.ca/data/en/dataset/5b401fa0-6c29-57f0-b3d5-749f301d829d>
