# Figure 2 chip-task visual assets

## Objective

Prepare real-image assets for the ForeTac method diagram: synchronized global/wrist camera inputs, project-page execution frames, and tactile RGB images overlaid with marker displacement in the style of Reactive Diffusion Policy Figure 3.

## Data audit

- Local usable chip dataset: `/home/chenshuai/Desktop/260709_v8j_jiashupian/jiashupian_0709/`.
- Ten episodes and 5,262 total frames were found.
- `episode_1.hdf5` contains synchronized `global`, `wrist`, left/right tactile RGB, and left/right `marker_offset` arrays.
- The HDF5 source has no acquisition timestamps. Assets therefore use episode/frame identifiers and do not claim exact wall-clock synchronization.
- The five-task TacVAE source configuration points to a larger 2026-07-10 dataset on an unavailable removable drive. The local 2026-07-09 episode was selected because it matches the chip task and the project-page demonstration.

## Rendering contract

- Six Global/Wrist pairs are exported from identical HDF5 indices.
- Twenty tactile frames span approach, grasp, transfer, and placement.
- Marker displacement is measured relative to the mean of the first 20 frames.
- The left tactile sensor was selected because its 9 x 9 marker pattern is clearer than the right sensor in this episode.
- Yellow arrows with a dark edge are drawn on the synchronized tactile RGB image, matching the visual language of RDP Figure 3.
- All twenty images use one shared `45x` display magnification. This is explicitly metadata-recorded and preserves relative comparisons across frames.

## Figure-design conclusion

Use real images only at semantic endpoints: observed RGB/touch inputs, decoded tactile outputs, and final robot execution. Keep latent grids, tokens, learned queries, and diffusion states abstract. In panel (c), compact contact-state icons remain clearer than five tiny photographs.

Observed tactile overlays must not be labeled as model predictions or reconstructions. The final PPT should replace prediction endpoints with actual decoded Foresight/TacVAE results while retaining the same visual style.
