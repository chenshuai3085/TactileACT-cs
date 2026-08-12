# Figure 2 chip-task image assets

## Sources

- Synchronized training observations: `episode_1.hdf5` from the local 2026-07-09 V8J potato-chip dataset.
- Execution images: the project-page source video `chip_real_raw.mp4`.
- Tactile overlays: left tactile RGB with the synchronized `9 x 9 x 2` marker displacement field.

Frame indices are used because the HDF5 file has no acquisition timestamp. Global RGB, wrist RGB, tactile RGB, and marker displacement with the same HDF5 frame index are synchronized.

## Files

- `camera_pairs/`: six synchronized Global/Wrist pairs.
- `marker_rdp/`: twenty text-free tactile RGB images with RDP-style yellow marker-displacement arrows.
- `execution/`: four uncropped project-page source-video frames.
- `execution_clean/`: the same frames cropped to remove the page-video title and speed badge.
- `*_contact_sheet.png`: selection sheets; do not use these directly in the paper.
- `metadata.json`: sources, frame indices, shared arrow-display scale, and displacement statistics.
- `asset_index.csv`: compact camera and tactile file index.

## Recommended Figure 2 placement

### Panel (a), TacVAE

- Input history: stack three RDP overlays, preferably frames 140, 163, and 187.
- Keep `mu`, `log variance`, sampling, and latent grids abstract.
- Reconstruction endpoint: use one RDP-style overlay only if it is an actual decoder reconstruction. Do not present an observed frame as a reconstructed result.

### Panel (b), Foresight

- Before Visual Encoder: use the synchronized Global RGB and Wrist RGB pair from frame 160 or 200.
- Before Frozen TacVAE: use one to three RDP overlays from the same temporal neighborhood.
- Keep visual tokens, tactile tokens, queries, and latent trajectory abstract.
- The three `Predicted contact evolution` images should be actual decoded model predictions. The observed RDP overlays in this asset pack are visual-style references or ground truth; do not label them as predictions.

### Panel (c), Scorer

- Keep the five compact contact-state icons. Five tiny tactile photographs would make this panel noisy and hard to read.

### Inference panel

- Keep predicted tactile futures as model-output visualizations.
- At `Execute and replan`, use two clean execution frames, preferably 6.0 s and 10.0 s, connected by a short time arrow.

## Rendering note

All twenty marker overlays share one `45x` display magnification so relative marker-motion magnitudes remain comparable. The scale is a visualization magnification, not a literal pixel-to-pixel arrow scale.
