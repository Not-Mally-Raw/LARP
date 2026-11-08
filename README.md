# LARP Research Observatory

A single-file, offline JavaScript interface for exploring the Ariel project’s catalog and archived analysis artifacts. The experience is designed as one evidence-first observatory, not a collection of tools that hand users between separate apps.

## Open it

Open `index.html` in a current browser with WebGL enabled. There is no build step, package installation, local server, or external script. The plots, data, WebGL scene, controls, and styles are embedded in the HTML file. The interface requests Space Grotesk and DM Sans from Google Fonts when online, then falls back to local system fonts when offline; ESA reference links open only when clicked.

## Product idea

The interface has four jobs:

1. Let a researcher orient themselves in the full 1,100-system catalog.
2. Make it easy to move from a population view to an individual planet dossier.
3. Show measured visit-level evidence and model outputs only where saved files exist.
4. Make missing coverage, stale outputs, and conflicting artifacts visible in the same workflow.

The 3D orbital scene is an interactive schematic for wayfinding and scientific storytelling. Its ring arcs, dust, and planet surface treatment are illustrative effects, not inferred properties of any catalog system. It is not a ray tracer, a radiative-transfer calculation, or a scale drawing. The population plot uses catalog metadata only; it does not imply that 1,100 visits were processed.

## Views and interactions

- **Observatory:** research status, data-coverage counts, an illustrative WebGL planet/orbit scene, an intuitive catalog scatter preview, and a labeled benchmark spectrum preview.
- **Planet catalog:** search by planet/star ID; filter by stellar class, holdout/fold, geometry, QC, or spectrum; page through all 1,100 systems; export the catalog CSV.
- **Planet dossier:** show catalog metadata for every ID, then add archived geometry diagnostics only for systems with saved C05 rows. Show the retrieval only for the target that has a retrieval artifact.
- **Spectrum lab:** toggle the saved Stage 5 WLS, Stage 6 FWL, and Stage 8 calibrated curves for Planet 1010375142; hover for channel values; export all 283 channels and uncertainties.
- **Transit theater:** compare AIRS and FGS1 as separate relative-depth plots, with a robust out-of-transit baseline, archived T₁₄ window, and explicitly display-only running median; inspect a dedicated geometry-fit slide and export the plotted depth series. No transit fit curve is fabricated.
- **Population map:** click or keyboard-select catalog points to open dossiers. Axes span the observed stellar-temperature and orbital-period ranges; color denotes host-star spectral class, while controls filter the development/holdout partition.
- **How Ariel works:** follow an interactive optical-path schematic and a five-step mission-to-measurement guide, from transit and telescope optics through parallel FGS/AIRS channels to a conceptual transmission spectrum. The diagrams are explanatory, not a ray trace or scale drawing; the spectrum sketch is not a project result or molecular detection. The page distinguishes the mission instrument suite from this project’s FGS1 + AIRS-CH0 benchmark channels.
- **Evidence ledger:** inspect the scope limits, source paths, SHA-256 hashes, and known differences between duplicate artifacts; export the saved 30-row geometry table.

Drag the orbital scene to rotate it and use the scroll wheel to zoom. Use the top search and press Enter to open a planet dossier. Press `?` to open the in-app usage guide.

## What the bundled snapshot contains

- 1,100 catalog systems: 988 assigned to development folds and 112 marked as internal holdout.
- 30 archived Stage 4 geometry records; 6 have a saved QC pass.
- Full saved spectra across 283 channels and a single-target atmospheric retrieval for Planet 1010375142.
- A target transit time series and a catalog-level temperature/period view.

A “catalog only” row means no visit-level analysis artifact is bundled for that ID. It does not mean the planet failed.

## Scientific and provenance boundaries

The repository note `docs/03_reproducibility_and_result_status.md` states that saved notebook outputs predate edited notebook source. Therefore this UI labels them as historical and does not present them as a rerun of the edited notebooks. Old OOF/vault scores, coverage rates, confidence intervals, ensemble gains, population anomaly rates, and other withdrawn claims are intentionally excluded.

Several duplicate C05/C06/C07/C10 artifacts differ across `contracts/` and `results/`. The app names the exact file it plots, lists alternate mirrors and their hashes, and reports the maximum per-channel depth differences. The standalone target C05 row also disagrees with the target row in the 30-planet batch summary; the app uses the batch row in cohort summaries and marks the target geometry discrepancy unresolved.

The displayed target spectra come from distinct saved files without a shared run identifier. Their overlay is an archived comparison, not proof that all stages were produced by one reproducible execution. The Stage 8 uncertainty curve is a saved output, not a validated held-out population calibration result.

The gas values are saved single-target retrieval outputs without posterior intervals. The interface does not show molecular band markers, a pressure–temperature profile, or ray tracing because the bundled artifacts do not include the opacity/cross-section inputs or validated radiative-transfer results those views would require.

## Design references

The refreshed palette and planet-forward editorial composition take inspiration from the [Behance Galaxy Space concept](https://www.behance.net/gallery/164659865/Galaxy-Space-solar-system-Concept-UI). The five-step “How Ariel works” sequence is also inspired by the user-provided infographic, while correcting its serial FGS-to-AIRS depiction: Ariel’s FGS and AIRS channels are separate branches from the telescope assembly. The interface retains its own evidence-first structure and uses no ESA logo or copied mission artwork. The visual hierarchy also takes cues from official mission-science sites: NASA Webb’s searchable, filterable image archive separates the science content from its explanatory context, while ESA’s Ariel page leads with mission purpose and a compact set of verified facts. The observatory applies those principles to this local research snapshot: clear scope labels, evidence-first navigation, and restrained mission-console styling. It does not use NASA or ESA marks and is not affiliated with or endorsed by either agency.

- [NASA Webb image archive](https://science.nasa.gov/mission/webb/multimedia/images/)
- [ESA Ariel mission overview](https://www.esa.int/Science_Exploration/Space_Science/Ariel)
- [ESA Ariel instruments](https://www.esa.int/Science_Exploration/Space_Science/Ariel/Ariel_s_instruments)
- [ESA transmission spectroscopy explainer](https://www.esa.int/Science_Exploration/Space_Science/Ariel/Studying_exoplanet_atmospheres_with_Ariel)

## Implementation

`index.html` contains the complete application: semantic HTML, responsive CSS, vanilla JavaScript, inline JSON data, SVG charts, and a native WebGL orbital scene with visibly labeled illustrative effects. There is no React, Three.js, Plotly, Streamlit, or backend. CSV export actions create files locally in the browser.
