# Submission checklist

Run this top-to-bottom before submitting the manuscript anywhere.

## Content

- [ ] Table IV's `PENDING-HW` cells filled *or* the table explicitly limited
      to desktop-verified rows (do not submit latent PENDING cells as
      measured values)
- [ ] Abstract/intro numbers re-traced after any table regeneration
      (`training/scripts/build_tables.py` then a VERIFICATION-style pass)
- [ ] Deployed-model vs KFall-only model numbers not conflated (abstract,
      results §V-F) — the conflation error found & fixed in
      `manuscript/VERIFICATION.md` §2
- [ ] Limitations section covers: false-alarm trade-off, dataset age/demo
      skew (datasets ≠ elderly falls), waist-only placement, pending hardware
      numbers, single-trial replay scope
- [ ] Operating-point statement matches what the firmware actually ships
      (default 0.95 / k=2; model-card recommendation 0.7 / k=4) — both
      documented, one table row each

## Compliance

- [ ] All five dataset citations present and in the bibliography (Sucerquia
      2017; Musci 2018 & 2020; Yu 2021; Saleh 2020; Casilari 2017)
- [ ] No raw dataset content submitted (figures are aggregates/figures only)
- [ ] Author list, affiliations, IDs match the team's agreement; venue
      template is the conference's current one (do not reuse last year's)
- [ ] Anonymised/submitted-with-authors variant as the CFP requires
- [ ] Artifacts link (this repository) included; repo publicly reachable;
      README renders correctly (Wokwi steps work as written)

## Repository-side (this repo)

- [ ] `docs/RESULTS_PROVENANCE.md` has a row for every number in the PDF
- [ ] `experiments/hardware_measurement/` either filled with real values or
      referenced as "measurement protocol defined, execution in progress"
- [ ] Demo: Wokwi replay runs from a clean clone following `docs/WOKWI.md`
- [ ] No secrets/credentials anywhere in history (`git log -p | grep -i token|api|key|secret`)
- [ ] Clean `git status`; tags: conference submission version (e.g. `v1.0-conference`)
